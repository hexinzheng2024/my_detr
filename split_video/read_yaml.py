import yaml

# 打开并读取 YAML 文件
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# 访问 YAML 文件中的参数
print(config)

# 示例：访问特定的参数
database_host = config['database']['host']
database_port = config['database']['port']
logging_level = config['logging']['level']

print(f"Database Host: {database_host}")
print(f"Database Port: {database_port}")
print(f"Logging Level: {logging_level}")
