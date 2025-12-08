import os

dev_databases_path = 'database/dev_databases'
dev_json_path = 'data/dev.json'

# LLM 配置：默认读取环境变量，未设置时使用安全默认值
model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')  # 你可改为真实模型名
api = os.getenv('OPENAI_API_KEY', '')             # 必须设置有效 API Key
base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')