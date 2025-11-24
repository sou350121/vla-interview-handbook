#!/bin/bash
# GitHub Personal Access Token 配置脚本

echo "=========================================="
echo "GitHub Personal Access Token 配置"
echo "=========================================="
echo ""
echo "请按照以下步骤获取您的 Personal Access Token："
echo "1. 访问：https://github.com/settings/tokens"
echo "2. 点击 'Generate new token' -> 'Generate new token (classic)'"
echo "3. 设置名称和过期时间"
echo "4. 勾选 'repo' 权限"
echo "5. 生成并复制 token"
echo ""
read -p "请输入您的 GitHub Personal Access Token: " TOKEN

if [ -z "$TOKEN" ]; then
    echo "错误：Token 不能为空"
    exit 1
fi

# 方法 1：使用 credential helper 存储
echo "https://sou350121:${TOKEN}@github.com" | git credential approve

echo ""
echo "✓ 凭据已配置完成！"
echo ""
echo "现在可以运行以下命令推送代码："
echo "  git push origin main"
echo ""

