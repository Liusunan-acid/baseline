#!/bin/bash

# 1. 显示当前状态
echo "=== 正在检查状态... ==="
git status

# 2. 添加所有修改 (包括新文件和删除的文件)
echo "=== 正在添加文件 (git add)... ==="
git add .

# 3. 提交修改
# 如果你在运行脚本时没有写备注，默认使用 "Auto update"
msg="${1:-Auto update}"
echo "=== 正在提交 (git commit)... 备注: $msg ==="
git commit -m "$msg"

# 4. 推送到 GitHub
echo "=== 正在推送到 GitHub... ==="
git push

echo "=== ✅ 全部完成! ==="