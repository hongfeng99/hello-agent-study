from dotenv import load_dotenv
import os
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE") or None

print("NEO4J_URI =", uri)
print("NEO4J_USERNAME =", username)
print("NEO4J_DATABASE =", database)

if not uri:
    raise ValueError("没有读取到 NEO4J_URI，请检查 .env 是否在项目根目录")
if not password:
    raise ValueError("没有读取到 NEO4J_PASSWORD，请检查 .env 是否配置正确")

driver = GraphDatabase.driver(uri, auth=(username, password))

try:
    print("\n正在验证 Neo4j 连接...")
    driver.verify_connectivity()
    print("Neo4j 连接验证成功")

    session_kwargs = {}
    if database:
        session_kwargs["database"] = database

    with driver.session(**session_kwargs) as session:
        result = session.run(
            """
            MERGE (n:HelloAgentsTest {name: $name})
            RETURN n.name AS name
            """,
            name="write_test"
        )
        record = result.single()
        print("Neo4j 写入成功：", record["name"])

finally:
    driver.close()