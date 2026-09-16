#!/bin/bash
# setup.sh: NISCコンパイラのセットアップスクリプト
#
# 使い方: source ./setup.sh

# 仮想環境がなければ作る
if [ ! -d ".compiler" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .compiler
else
    echo "Virtual environment already exists, skipping..."
fi

# 仮想環境を有効化
source .compiler/bin/activate

# パッケージをインストール
pip install -e . -q
pip install pycparser xdsl networkx -q

echo "Setup complete!"
echo "Virtual environment is now active."