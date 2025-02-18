import os
import sys
import json
import requests
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import Neo4jVector
from py2neo import Graph
from zhipuai import ZhipuAI
import re
import heapq
from collections import defaultdict

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)

# **确保 BASE_DIR 在最前面定义**
if getattr(sys, 'frozen', False):  # 检测是否为 PyInstaller 打包环境
    BASE_DIR = sys._MEIPASS  # PyInstaller 解压后的临时目录
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 普通 Python 运行环境

# 读取配置文件
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
else:
    config = {}

# 设定 input_data 文件夹路径
INPUT_DATA_FOLDER = os.path.join(BASE_DIR, "input_data")
if not os.path.exists(INPUT_DATA_FOLDER):
    os.makedirs(INPUT_DATA_FOLDER)  # 确保文件夹存在

# 设置 OpenAI 反向代理
TARGET_OPENAI_PROXY = config.get("openai_proxy", "http://default-proxy-url.com")  # 默认值可自行修改

# 连接到 Neo4j 数据库
neo4j_config = config.get("neo4j", {})
NEO4J_URL = neo4j_config.get("url", "bolt://localhost:7687")
NEO4J_USERNAME = neo4j_config.get("username", "neo4j")
NEO4J_PASSWORD = neo4j_config.get("password", "neo4j")

graph = Graph(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# 读取智谱 API Key
zhipu_config = config.get("zhipu", {})
ZHIPU_API_KEY = zhipu_config.get("api_key", "")

# 初始化智谱客户端
client = ZhipuAI(api_key=ZHIPU_API_KEY)

def flatten_properties(properties):
    """将嵌套字典转换为 JSON 字符串"""
    for key, value in properties.items():
        if isinstance(value, dict):  # 如果是字典，转换为字符串存储
            properties[key] = json.dumps(value, ensure_ascii=False)
    return properties

# 添加自动创建向量索引的方法
def ensure_vector_index():
    """为所有预定义的节点类型创建两个 Neo4j 向量索引：basic_embedding 和 full_embedding"""
    try:
        # **1. 从 config.json 读取预定义的节点类型**
        predefined_labels = config.get("predefined_labels", ["entity"])

        if not isinstance(predefined_labels, list) or not predefined_labels:
            print("❌ 配置文件中的 `predefined_labels` 无效，使用默认值。")
            predefined_labels = ["school_uniform", "student", "teacher"]

        print(f"📌 预定义的节点类型: {predefined_labels}")

        # **2. 检查 GDS 插件是否安装**
        gds_check_query = "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'gds.' RETURN name LIMIT 1"
        gds_installed = bool(graph.run(gds_check_query).data())

        if not gds_installed:
            print("❌ Neo4j Graph Data Science (GDS) 插件未安装，向量索引可能无法使用！")
            return

        # **3. 检查已有索引**
        check_query = "SHOW INDEXES YIELD name"
        existing_indexes = [index["name"] for index in graph.run(check_query).data()]

        # **4. 遍历所有预定义的节点类型，创建两个索引**
        for label in predefined_labels:
            for embedding_type in ["basic", "full"]:
                index_name = f"vector_index_{label}_{embedding_type}"
                embedding_field = f"{embedding_type}_embedding"

                if index_name not in existing_indexes:
                    print(f"🔍 向量索引 `{index_name}` 不存在，正在创建...")
                    create_index_query = f"""
                    CREATE VECTOR INDEX {index_name} FOR (n:`{label}`) ON (n.{embedding_field})
                    OPTIONS {{
                      indexConfig: {{
                        `vector.dimensions`: 2048,
                        `vector.similarity_function`: "cosine"
                      }}
                    }};
                    """
                    graph.run(create_index_query)
                    print(f"✅ 向量索引 `{index_name}` 创建成功！")
                else:
                    print(f"✅ 向量索引 `{index_name}` 已存在，无需创建。")

    except Exception as e:
        print(f"❌ 创建向量索引失败：{e}")

# 删除全部向量索引
@app.route('/neo4j/delete_vector_indexes', methods=['POST'])
def delete_all_vector_indexes_api():
    """提供 API 端点以删除 Neo4j 所有向量索引"""
    try:
        # **1. 获取所有索引**
        check_query = "SHOW INDEXES YIELD name"
        existing_indexes = [index["name"] for index in graph.run(check_query).data()]

        # **2. 过滤出所有 `vector_index_` 开头的索引**
        vector_indexes = [idx for idx in existing_indexes if idx.startswith("vector_index_")]

        if not vector_indexes:
            print("⚠️ 没有向量索引需要删除。")
            return jsonify({"status": "warning", "message": "没有向量索引需要删除"}), 200

        # **3. 逐个删除索引**
        for index_name in vector_indexes:
            print(f"🗑 正在删除向量索引 `{index_name}`...")
            drop_query = f"DROP INDEX {index_name}"
            graph.run(drop_query)
            print(f"✅ 向量索引 `{index_name}` 已成功删除！")

        return jsonify({"status": "success", "message": f"已删除 {len(vector_indexes)} 个向量索引"}), 200

    except Exception as e:
        print(f"❌ 删除向量索引失败：{e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# 定义 embedding 计算方法
class ZhipuAIEmbeddings:
    def embed_query(self, text):
        print(f"🧠 计算 embedding: {text}")
        try:
            API_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
            HEADERS = {
                "Authorization": f"Bearer {ZHIPU_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "embedding-3",
                "input": [text]  # 必须是列表
            }

            response = requests.post(API_URL, json=payload, headers=HEADERS)
            response_json = response.json()

            if "error" in response_json:
                print(f"❌ 智谱 API 错误: {response_json['error']['message']}")
                return None

            # 确保返回的数据格式正确
            if "data" not in response_json or not isinstance(response_json["data"], list):
                print(f"❌ API 返回格式错误: {response_json}")
                return None

            embedding = response_json["data"][0]["embedding"]
            if not embedding:
                print(f"❌ 无法提取嵌入向量: {response_json}")
                return None

            print(f"✅ 提取的 embedding 长度: {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"❌ 计算 embedding 出错: {e}")
            return None

embedding_model = ZhipuAIEmbeddings()

def read_file_content(file_path):
    """读取 JSON 或 TXT 文件的内容，并返回解析后的数据"""
    try:
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
                return {"name": os.path.basename(file_path), "content": content} if not content.startswith("{") else json.loads(content)
    except Exception as e:
        print(f"❌ 读取文件失败: {file_path}, 错误: {e}")
        return None

def reset_vector_index():
    """重新计算所有节点的 basic_embedding 和 full_embedding"""
    try:
        # **1. 删除所有节点的 embedding**
        graph.run("MATCH (n) REMOVE n.basic_embedding, n.full_embedding")
        print("✅ 已删除所有节点的向量数据")

        # **2. 重新计算所有节点的 embedding**
        query_nodes = "MATCH (n) RETURN n.name AS name, labels(n) AS labels, properties(n) AS properties"
        nodes = graph.run(query_nodes).data()

        for node in nodes:
            name = node.get("name", "Unnamed")
            labels = ", ".join(node.get("labels", []))  # 例如 "Person, Character"
            properties = node.get("properties", {})

            # **3. 计算 basic_embedding**
            alias = properties.get("alias", "")
            tag = properties.get("tag", "")

            # 如果 alias 和 tag 存在，则加入文本，否则忽略
            basic_text_parts = [f"名称: {name}"]
            if alias:
                basic_text_parts.append(f"别名: {alias}")
            if tag:
                basic_text_parts.append(f"标签: {tag}")

            basic_text = ", ".join(basic_text_parts)
            basic_embedding = embedding_model.embed_query(basic_text)

            if basic_embedding is None:
                print(f"⚠️ 无法计算 `{name}` 的 basic_embedding，跳过。")
                continue

            # **4. 计算 full_embedding**
            filtered_properties = {k: v for k, v in properties.items() if k not in ["basic_embedding", "full_embedding"]}
            full_text = f"名称: {name}, 类型: {labels}, 详细信息: {json.dumps(filtered_properties, ensure_ascii=False)}"
            full_embedding = embedding_model.embed_query(full_text)

            if full_embedding is None:
                print(f"⚠️ 无法计算 `{name}` 的 full_embedding，跳过。")
                continue

            # **5. 存入 Neo4j**
            update_query = """
            MATCH (n {name: $name})
            SET n.basic_embedding = $basic_embedding,
                n.full_embedding = $full_embedding
            """
            graph.run(update_query, name=name, basic_embedding=basic_embedding, full_embedding=full_embedding)

        print("✅ 全量向量化完成")
    except Exception as e:
        print(f"❌ 全量向量化失败: {e}")

def incremental_vectorize(node_names):
    """增量向量化新添加的节点，计算 basic_embedding 和 full_embedding"""
    for name in node_names:
        # **1. 查询节点属性**
        query = """
        MATCH (n {name: $name})
        RETURN labels(n) AS labels, properties(n) AS properties
        """
        node_data = graph.run(query, name=name).data()

        if not node_data:
            print(f"⚠️ 节点 `{name}` 不存在，跳过。")
            continue

        labels = ", ".join(node_data[0].get("labels", []))
        properties = node_data[0].get("properties", {})

        # **2. 计算 basic_embedding**
        alias = properties.get("alias", "")
        tag = properties.get("tag", "")

        basic_text_parts = [f"名称: {name}"]
        if alias:
            basic_text_parts.append(f"别名: {alias}")
        if tag:
            basic_text_parts.append(f"标签: {tag}")

        basic_text = ", ".join(basic_text_parts)
        basic_embedding = embedding_model.embed_query(basic_text)

        if basic_embedding is None:
            print(f"⚠️ 无法计算 `{name}` 的 basic_embedding，跳过。")
            continue

        # **3. 计算 full_embedding**
        filtered_properties = {k: v for k, v in properties.items() if k not in ["basic_embedding", "full_embedding"]}
        full_text = f"名称: {name}, 类型: {labels}, 详细信息: {json.dumps(filtered_properties, ensure_ascii=False)}"
        full_embedding = embedding_model.embed_query(full_text)

        if full_embedding is None:
            print(f"⚠️ 无法计算 `{name}` 的 full_embedding，跳过。")
            continue

        # **4. 更新 Neo4j**
        update_query = """
        MATCH (n {name: $name})
        SET n.basic_embedding = $basic_embedding,
            n.full_embedding = $full_embedding
        """
        graph.run(update_query, name=name, basic_embedding=basic_embedding, full_embedding=full_embedding)

    print("✅ 增量向量化完成")

# 创建关系
def create_relationships_for_nodes(node_names=None):
    """为增量或全量导入的节点创建关系"""
    try:
        relationships = config.get("create_relationship", [])
        if not relationships:
            print("⚠️ 配置文件中没有定义任何关系规则，跳过创建关系。")
            return

        if not node_names:
            query_nodes = "MATCH (n) RETURN n.name AS name"
            node_names = [record["name"] for record in graph.run(query_nodes).data()]

        if not node_names:
            print("⚠️ 没有找到需要处理的节点，跳过关系创建。")
            return

        for rule in relationships:
            if len(rule) != 3:
                continue

            source_label, relationship_type, target_label = rule
            print(f"🔗 处理关系: ({source_label}) -[:{relationship_type}]-> ({target_label})")

            # **动态获取目标字段名**
            query_source = f"""
            MATCH (s:`{source_label}`)
            WHERE s.name IN $node_names
            RETURN s.name AS source_name, s.`{target_label.lower()}` AS target_names
            """
            source_nodes = graph.run(query_source, node_names=node_names).data()

            for source_node in source_nodes:
                source_name = source_node["source_name"]
                target_names = source_node.get("target_names")

                if not target_names:
                    continue

                # 确保 target_names 是列表
                if not isinstance(target_names, list):
                    target_names = [target_names]

                for target_name in target_names:
                    query_target = f"""
                    MATCH (t:`{target_label}`)
                    WHERE t.name = $target_name OR $target_name IN t.alias
                    RETURN t.name AS target_name
                    """
                    target_nodes = graph.run(query_target, target_name=target_name).data()

                    if not target_nodes:
                        print(f"⚠️ 未找到 `{target_name}` 的匹配节点，跳过")
                        continue

                    for target_node in target_nodes:
                        create_relationship_query = f"""
                        MATCH (s:`{source_label}` {{name: $source_name}})
                        MATCH (t:`{target_label}` {{name: $target_name}})
                        MERGE (s)-[:`{relationship_type}`]->(t)
                        """
                        graph.run(create_relationship_query, source_name=source_name, target_name=target_node["target_name"])
                        print(f"✅ 关系已创建: ({source_name}) -[:{relationship_type}]-> ({target_node['target_name']})")

    except Exception as e:
        print(f"❌ 关系创建失败: {e}")

def convert_json_for_neo4j(data):
    """
    遍历 JSON，将所有对象 {} 和数组 [] 转换为字符串，以适应 Neo4j 存储
    """
    if isinstance(data, dict):
        return {k: convert_json_for_neo4j(v) for k, v in data.items()}
    elif isinstance(data, list):
        return json.dumps(data, ensure_ascii=False)  # 将列表转换为 JSON 字符串
    else:
        return data  # 其他类型（字符串、数字、布尔值）保持不变

# 全量导入
@app.route('/neo4j/full_import', methods=['POST'])
def full_import():
    """全量删除数据后重新导入，并创建关系、重新计算向量"""
    try:
        # **1. 删除所有节点和关系**
        print("🗑 删除所有节点和关系...")
        graph.run("MATCH (n) DETACH DELETE n")
        print("✅ 所有数据已删除")

        # **2. 读取 `input_data` 目录中的所有 JSON/TXT 文件**
        if not os.path.exists(INPUT_DATA_FOLDER):
            return jsonify({"status": "error", "message": "input_data 文件夹不存在"}), 400

        results, new_nodes = [], []

        for filename in os.listdir(INPUT_DATA_FOLDER):
            if filename.endswith((".json", ".txt")):
                file_path = os.path.join(INPUT_DATA_FOLDER, filename)
                data = read_file_content(file_path)
                if not data or "name" not in data:
                    print(f"⚠️ 跳过无效文件: {filename}")
                    continue

                node_name = data["name"]
                label = data.get("type", "GenericEntity")  # 默认标签
                # **转换 JSON 结构**
                properties = convert_json_for_neo4j(data)

                # **创建节点**
                properties = flatten_properties(properties)
                create_query = f"CREATE (n:`{label}` {{ {', '.join(f'{k}: ${k}' for k in properties)} }}) RETURN n"
                result = graph.run(create_query, **properties).data()
                results.append({"file": filename, "created_node": result})
                new_nodes.append(node_name)

        print(f"✅ 已导入 {len(new_nodes)} 个新节点")

        # **3. 创建所有关系**
        create_relationships_for_nodes()
        print("✅ 已创建所有关系")

        # **4. 删除所有向量数据**
        print("🗑 删除所有节点的向量数据...")
        graph.run("MATCH (n) REMOVE n.embedding")
        print("✅ 已删除所有向量数据")

        # **5. 重新计算所有节点的向量**
        reset_vector_index()
        print("✅ 全量向量化完成")

        return jsonify({"status": "completed", "deleted_old": True, "new_nodes": new_nodes, "details": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/neo4j/incremental_import', methods=['POST'])
def incremental_import():
    """增量导入数据，并自动创建关系和向量化"""
    if not os.path.exists(INPUT_DATA_FOLDER):
        return jsonify({"status": "error", "message": "input_data 文件夹不存在"}), 400

    results, new_nodes = [], []

    for filename in os.listdir(INPUT_DATA_FOLDER):
        if filename.endswith((".json", ".txt")):
            file_path = os.path.join(INPUT_DATA_FOLDER, filename)
            data = read_file_content(file_path)
            if not data or "name" not in data:
                continue

            node_name = data["name"]
            count = graph.run("MATCH (n {name: $name}) RETURN COUNT(n) AS count", name=node_name).evaluate()

            if count == 0:  # **如果数据库中不存在该节点，则创建**
                label = data.get("type", "GenericEntity")
                # **转换 JSON 结构**
                properties = convert_json_for_neo4j(data)

                properties = flatten_properties(properties)
                create_query = f"CREATE (n:`{label}` {{ {', '.join(f'{k}: ${k}' for k in properties)} }}) RETURN n"
                result = graph.run(create_query, **properties).data()
                results.append({"file": filename, "created_node": result})
                new_nodes.append(node_name)

    print(f"✅ 增量导入了 {len(new_nodes)} 个新节点")

    # **2. 为新增的节点创建关系**
    if new_nodes:
        create_relationships_for_nodes(new_nodes)
        print("✅ 已创建新增节点的关系")

        # **3. 为新增的节点进行增量向量化**
        incremental_vectorize(new_nodes)
        print("✅ 已完成新增节点的向量化")

    return jsonify({"status": "completed", "new_nodes": new_nodes, "details": results})

@app.route('/neo4j/import_file', methods=['POST'])
def import_file():
    """根据请求的文件名导入或更新数据"""
    try:
        data = request.get_json()
        filename = data.get("filename")

        if not filename:
            return jsonify({"status": "error", "message": "缺少参数 `filename`"}), 400

        file_path = os.path.join(INPUT_DATA_FOLDER, filename)

        if not os.path.exists(file_path):
            return jsonify({"status": "error", "message": f"文件 `{filename}` 不存在"}), 404

        # **1. 读取文件内容**
        file_data = read_file_content(file_path)
        if not file_data or "name" not in file_data:
            return jsonify({"status": "error", "message": f"文件 `{filename}` 格式无效"}), 400

        node_name = file_data["name"]
        label = file_data.get("type", "GenericEntity")  # 默认标签
        properties = {k: v for k, v in file_data.items() if k != "type"}

        # **2. 展平属性，防止嵌套字典错误**
        properties = flatten_properties(properties)

        # **3. 检查数据是否已存在**
        existing_count = graph.run("MATCH (n {name: $name}) RETURN COUNT(n) AS count", name=node_name).evaluate()

        if existing_count > 0:
            # **更新已有节点**
            update_query = f"""
            MATCH (n:`{label}` {{name: $name}})
            SET {', '.join(f'n.{k} = ${k}' for k in properties)}
            RETURN n
            """
            result = graph.run(update_query, name=node_name, **properties).data()
            action = "updated"
        else:
            # **创建新节点**
            create_query = f"""
            CREATE (n:`{label}` {{ {', '.join(f'{k}: ${k}' for k in properties)} }})
            RETURN n
            """
            result = graph.run(create_query, **properties).data()
            action = "created"

        # **4. 创建关系**
        create_relationships_for_nodes([node_name])

        # **5. 计算向量**
        incremental_vectorize([node_name])

        return jsonify({
            "status": "success",
            "action": action,
            "node": node_name,
            "details": result
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/neo4j/delete_all', methods=['POST'])
def delete_all_nodes():
    """删除 Neo4j 数据库中的所有节点及其关系"""
    try:
        graph.run("MATCH (n) DETACH DELETE n")
        return jsonify({"status": "success", "message": "All nodes and relationships deleted."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def split_into_sentences(text):
    """使用正则表达式将文本拆分为句子，支持中、英、日"""
    sentence_endings = r'(?<=[。！？.!?])\s*'  # 适用于中、英、日的句号、问号、感叹号
    sentences = re.split(sentence_endings, text)
    return [s.strip() for s in sentences if s.strip()]

def extract_query_message(messages):
    """提取 `<query_message>` 并拼接内容"""
    query_text = ""
    in_query_message = False

    for msg in messages:
        content = msg["content"]

        if "<query_message>" in content:
            in_query_message = True
            query_text = content.split("<query_message>")[-1]  # 取 <query_message> 后半部分

        elif "</query_message>" in content:
            query_text += "\n" + content.split("</query_message>")[0]  # 取 </query_message> 前半部分
            in_query_message = False
            break  # 结束拼接

        elif in_query_message:
            query_text += "\n" + content  # 继续拼接

    if query_text:
        query_text = f"<query_message>{query_text}</query_message>"

    return query_text if query_text else None

def extract_clean_query_message(query_message):
    """去除 `<query_message>` 标签，返回纯文本"""
    return query_message.replace("<query_message>", "").replace("</query_message>", "").strip()

def find_related_nodes_by_sentences(query_message):
    """对 query_message 进行清理后分句，并分别基于 basic_embedding 和 full_embedding 查询最相似的节点"""
    clean_query = extract_clean_query_message(query_message)  # 去掉 <query_message> 标签
    sentences = split_into_sentences(clean_query)  # 分句
    print(f"📝 分句结果: {sentences}")

    node_scores = defaultdict(float)

    # **查询所有 `vector_index_*_basic` 和 `vector_index_*_full` 索引**
    check_query = "SHOW INDEXES YIELD name"
    existing_indexes = [index["name"] for index in graph.run(check_query).data()]
    vector_indexes_basic = [idx for idx in existing_indexes if idx.endswith("_basic")]
    vector_indexes_full = [idx for idx in existing_indexes if idx.endswith("_full")]

    for sentence in sentences:
        query_embedding = embedding_model.embed_query(sentence)
        if query_embedding is None:
            print(f"⚠️ 无法计算 `{sentence}` 的 embedding，跳过。")
            continue

        # **存储当前句子的最大相似度**
        sentence_max_scores = {}

        # **查询 `basic_embedding` 相似度**
        for index_name in vector_indexes_basic:
            query = f"""
            CALL db.index.vector.queryNodes('{index_name}', 10, $embedding)
            YIELD node, score
            RETURN node.name AS name, labels(node) AS labels, score
            """
            result = graph.run(query, embedding=query_embedding).data()

            for record in result:
                node_key = (record["name"], tuple(record["labels"]))
                sentence_max_scores[node_key] = max(sentence_max_scores.get(node_key, 0), record["score"])

        # **查询 `full_embedding` 相似度**
        for index_name in vector_indexes_full:
            query = f"""
            CALL db.index.vector.queryNodes('{index_name}', 10, $embedding)
            YIELD node, score
            RETURN node.name AS name, labels(node) AS labels, score
            """
            result = graph.run(query, embedding=query_embedding).data()

            for record in result:
                node_key = (record["name"], tuple(record["labels"]))
                sentence_max_scores[node_key] = max(sentence_max_scores.get(node_key, 0), record["score"])

        # **筛选相似度 >= 0.6 的节点**
        filtered_nodes = {k: v for k, v in sentence_max_scores.items() if v >= 0.6}

        # **取当前句子的前三个最相似节点**
        top_sentence_nodes = heapq.nlargest(3, filtered_nodes.items(), key=lambda x: x)

        # **累加得分**
        for (node_key, score) in top_sentence_nodes:
            node_scores[node_key] += score

    # **最终取全局得分最高的前三个节点**
    top_nodes = heapq.nlargest(3, node_scores.items(), key=lambda x: x)
    top_nodes = [{"name": name, "labels": labels} for (name, labels), _ in top_nodes]

    print(f"🎯 选出的前三个相关节点: {top_nodes}")

    return top_nodes

def get_relationship_types():
    """从 Neo4j 查询所有唯一的关系类型"""
    try:
        query = "MATCH ()-[r]->() RETURN DISTINCT type(r) AS relationship_type"
        result = graph.run(query).data()

        # 提取所有关系类型并去重
        relationship_types = sorted(set(record["relationship_type"] for record in result if record["relationship_type"]))

        print(f"📌 发现的关系类型: {relationship_types}")
        return relationship_types

    except Exception as e:
        print(f"❌ 获取关系类型失败: {e}")
        return []

def get_combined_node_content(nodes):
    """
    获取多个相关节点的详细信息，并外扩一层获取其直接关联的节点，去除 embedding 字段。
    """
    content_parts = []
    processed_nodes = set()  # 记录已处理的节点，避免重复

    # **1. 获取所有关系类型**
    relationship_types = get_relationship_types()
    if not relationship_types:
        print("⚠️ 没有找到任何关系类型，跳过外扩查询。")
        relationship_filter = ""
    else:
        relationship_filter = "|:".join(f"`{rel}`" for rel in relationship_types)  # 构造 `|:` 连接的关系字符串

    for node in nodes:
        node_name = node.get("name", "未知节点")

        # **2. 获取当前节点的所有属性**
        query = """
        MATCH (n {name: $name})
        RETURN properties(n) AS props
        """
        result = graph.run(query, name=node_name).data()

        if not result:
            continue

        properties = result[0]["props"]

        # **3. 过滤掉 `basic_embedding` 和 `full_embedding`**
        properties = {k: v for k, v in properties.items() if k not in ["basic_embedding", "full_embedding"]}

        # **4. 格式化当前节点内容**
        content = f"🔍 [RAG] 资料库自动提取: {node_name}"
        for key, value in properties.items():
            content += f"\n- {key}: {value}"

        content_parts.append(content.strip())
        processed_nodes.add(node_name)  # 标记当前节点已处理

        # **5. 查询当前节点的直接关联节点**
        if relationship_filter:
            query = f"""
            MATCH (n {{name: $name}})-[:{relationship_filter}*1]-(related)
            RETURN DISTINCT related.name AS related_name, properties(related) AS related_props
            """
            related_nodes = graph.run(query, name=node_name).data()

            for related_node in related_nodes:
                related_name = related_node["related_name"]
                if related_name in processed_nodes:
                    continue  # 避免重复添加

                related_props = {k: v for k, v in related_node["related_props"].items() if k not in ["basic_embedding", "full_embedding"]}

                # **6. 格式化关联节点内容**
                related_content = f"🔗 关联节点: {related_name}"
                for key, value in related_props.items():
                    related_content += f"\n- {key}: {value}"

                content_parts.append(related_content.strip())
                processed_nodes.add(related_name)  # 标记关联节点已处理

    return "\n\n".join(content_parts)


def replace_rag_data(messages, combined_content):
    """
    遍历聊天记录，替换 `[RAG_data]` 标记，确保 `embedding` 不会被发送给 AI。
    """
    updated_messages = []
    for msg in messages:
        if "[RAG_data]" in msg["content"]:
            msg["content"] = msg["content"].replace("[RAG_data]", combined_content)
            print(f"✅ `[RAG_data]` 已替换: {msg['content']}")

        updated_messages.append(msg)

    return updated_messages

@app.route('/v1/<path:endpoint>', methods=['POST', 'GET'])
def proxy_request_with_rag(endpoint):
    """代理 OpenAI 兼容 API 请求，并查找 `<query_message>` 相关的最相似节点，增强聊天数据"""
    try:
        # **1. 读取 OpenAI 代理 URL**
        openai_proxy = config.get("openai_proxy", "http://default-openai-proxy.com")

        # **2. 获取前端传入的 API Key**
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "message": "缺少 API Key"}), 401

        # **3. 处理 `GET /v1/models` 请求**
        if request.method == "GET" and endpoint == "models":
            url = f"{openai_proxy}/v1/models"
            headers = {"Authorization": auth_header}

            response = requests.get(url, headers=headers)
            return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type', 'application/json'))

        # **4. 处理聊天请求**
        if endpoint == "chat/completions" and request.method == 'POST':
            data = request.get_json()
            messages = data.get("messages", [])

            if not messages:
                return jsonify({"status": "error", "message": "聊天记录为空"}), 400

            # **提取 `<query_message>` 并查找最相似的节点**
            query_message = extract_query_message(messages)
            related_nodes = None  # 用于存储查询到的相关节点

            if query_message:
                print(f"\n🔍 **提取的 `<query_message>` 内容:**\n{query_message}\n")
                related_nodes = find_related_nodes_by_sentences(query_message)  # **替换 `find_related_nodes`**

            # **如果找到 `[RAG_data]`，替换为最相似节点的内容**
            if related_nodes:
                combined_content = get_combined_node_content(related_nodes)
                messages = replace_rag_data(messages, combined_content)

            # **7. 转发请求到 OpenAI 代理**
            url = f"{openai_proxy}/v1/chat/completions"
            headers = {key: value for key, value in request.headers.items() if key.lower() != 'host'}
            headers["Authorization"] = auth_header  # **使用前端传入的 API Key**

            is_stream = data.get("stream", False)

            if is_stream:
                response = requests.post(url, json={"messages": messages, **data}, headers=headers, stream=True)
                return Response(stream_with_context(response.iter_content(chunk_size=1024)), content_type=response.headers['Content-Type'])
            else:
                response = requests.post(url, json={"messages": messages, **data}, headers=headers)
                return Response(response.content, status=response.status_code, content_type=response.headers['Content-Type'])

        # **8. 处理其他 OpenAI 兼容 API 请求**
        url = f"{openai_proxy}/v1/{endpoint}"
        headers = {"Authorization": auth_header}

        if request.method == "GET":
            response = requests.get(url, headers=headers)
        else:
            response = requests.request(request.method, url, headers=headers, json=request.get_json())

        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type', 'application/json'))

    except Exception as e:
        print(f"❌ 代理请求失败: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

flask_config = config.get("flask", {})
FLASK_PORT = flask_config.get("port", 8081)  # 默认端口 8081
FLASK_DEBUG = flask_config.get("debug", False)  # 默认关闭 debug 模式

if __name__ == '__main__':
    # 确保 Neo4j 向量索引已创建
    print("🔍 确保 Neo4j 向量索引已创建...")
    ensure_vector_index()

    # 启动 Flask 服务器
    print(f"🚀 启动 Flask 服务器，监听端口 {FLASK_PORT}，Debug 模式: {FLASK_DEBUG}")
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=FLASK_DEBUG)
