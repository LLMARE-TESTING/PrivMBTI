/**
 * 将 questions_35.json 中的题目信息（text, context, options 等）
 * 补充到 data 目录下所有用户 result.json 的 answers.items 字段中。
 *
 * 用法：node enrich-answers.js
 */

const fs = require('fs');
const path = require('path');

// --- 路径配置 ---
const DATA_DIR = path.join(__dirname, 'data');
const QUESTIONS_FILE = path.join(__dirname, 'questions_35.json');

// --- 加载题目文件 ---
const questionsData = JSON.parse(fs.readFileSync(QUESTIONS_FILE, 'utf-8'));

// 建立 questionId -> question 的索引
const questionById = {};
for (const q of questionsData.questions) {
  questionById[q.id] = q;
}

console.log(`已加载 ${Object.keys(questionById).length} 道题目`);

// --- 遍历所有用户目录 ---
const entries = fs.readdirSync(DATA_DIR, { withFileTypes: true });

let processedCount = 0;
let skippedCount = 0;

for (const entry of entries) {
  if (!entry.isDirectory()) continue;
  if (!entry.name.startsWith('participant-no-')) continue;

  const resultPath = path.join(DATA_DIR, entry.name, 'result.json');
  if (!fs.existsSync(resultPath)) {
    console.log(`  [跳过] ${entry.name} — 未找到 result.json`);
    skippedCount++;
    continue;
  }

  const result = JSON.parse(fs.readFileSync(resultPath, 'utf-8'));

  // 兼容两种 answers 格式：
  //   A) answers 是对象，items 在 result.answers.items 中
  //   B) answers 直接就是数组
  let items;
  if (result.answers && Array.isArray(result.answers.items)) {
    items = result.answers.items;
  } else if (Array.isArray(result.answers)) {
    items = result.answers;
  } else {
    console.log(`  [跳过] ${entry.name} — answers 格式无法识别`);
    skippedCount++;
    continue;
  }

  let enrichedCount = 0;
  for (const item of items) {
    const q = questionById[item.questionId];
    if (!q) {
      console.log(`  [警告] ${entry.name} — questionId=${item.questionId} 在 questions_35.json 中未找到`);
      continue;
    }

    // 补充题目信息（不覆盖已有字段）
    if (!item.question) {
      item.question = q.text;
      enrichedCount++;
    }
    if (!item.context && q.context) {
      item.context = q.context;
    }
    if (!item.options) {
      item.options = q.options;
    }
    // 补充 explanation（知识题）
    if (!item.explanation && q.explanation) {
      item.explanation = q.explanation;
    }
    // 补充 dir（计分方向）
    if (item.dir === undefined && q.dir !== undefined) {
      item.dir = q.dir;
    }
  }

  // 写回文件
  fs.writeFileSync(resultPath, JSON.stringify(result, null, 2), 'utf-8');
  console.log(`  [完成] ${entry.name} — 补充了 ${enrichedCount} 个字段`);
  processedCount++;
}

console.log(`\n处理完成：${processedCount} 个用户已更新，${skippedCount} 个跳过`);
