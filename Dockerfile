FROM python:3.11-slim

WORKDIR /app

# システム依存パッケージ
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# uv インストール
RUN pip install uv

# 依存パッケージのインストール
COPY pyproject.toml .
RUN uv pip install --system -e .

# ソースコードのコピー
COPY . .

# デフォルトコマンド
CMD ["python", "-c", "print('Urban Structure Analysis System Ready')"]

