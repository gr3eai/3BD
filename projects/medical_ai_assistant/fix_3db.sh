#!/bin/bash
echo "🔧 إصلاح نظام 3DB AI..."

# 1. تحديث pip
pip install --upgrade pip

# 2. تثبيت المكتبات المطلوبة
pip install together python-dotenv requests

# 3. إنشاء بيانات تجريبية
mkdir -p ~/.3db/sample_data
cat > ~/.3db/sample_data/medical.jsonl << 'SAMPLE'
{"messages": [{"role": "user", "content": "عندي صداع"}, {"role": "assistant", "content": "خذ قسطاً من الراحة"}]}
{"messages": [{"role": "user", "content": "ألم في الظهر"}, {"role": "assistant", "content": "جرب تمارين الإطالة"}]}
SAMPLE

# 4. تحديث وظيفة Kimi Chat
sed -i "s/moonshotai\/kimi-k2-instruct-0905/moonshot-v1-32k/g" ~/.bashrc_3db

# 5. إعادة تحميل النظام
source ~/.bashrc

echo "✅ الإصلاح اكتمل! جرب الأوامر التالية:"
echo "   kimi_chat 'مرحباً'"
echo "   lora_finetune ~/.3db/sample_data/medical.jsonl"
