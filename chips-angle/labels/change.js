const fs = require('fs');
const path = require('path');

// ====== 配置 ======
const LABEL_DIR = './chips-angle/labels/val';   // 你的 txt 标注目录
const BACKUP = false;            // 是否备份原文件（.bak）
// ==================

function deg2rad(deg) {
  return deg * Math.PI / 180;
}

fs.readdirSync(LABEL_DIR).forEach(file => {
  if (!file.endsWith('.txt')) return;

  const filePath = path.join(LABEL_DIR, file);
  const raw = fs.readFileSync(filePath, 'utf8');

  if (!raw.trim()) return;

  const lines = raw.split(/\r?\n/);
  const newLines = [];

  let modified = false;

  for (const line of lines) {
    if (!line.trim()) continue;

    const parts = line.trim().split(/\s+/);
    if (parts.length !== 6) {
      console.warn(`[跳过] ${file} 非 OBB 格式: ${line}`);
      newLines.push(line);
      continue;
    }

    let angleDeg = parseFloat(parts[5]);
    if (isNaN(angleDeg)) {
      console.warn(`[跳过] ${file} angle 非数字: ${line}`);
      newLines.push(line);
      continue;
    }

    const angleRad = deg2rad(angleDeg);
    parts[5] = angleRad.toFixed(6);

    newLines.push(parts.join(' '));
    modified = true;
  }

  if (modified) {
    if (BACKUP) {
      fs.copyFileSync(filePath, filePath + '.bak');
    }
    fs.writeFileSync(filePath, newLines.join('\n'), 'utf8');
    console.log(`[OK] ${file}`);
  }
});

console.log('✔ 角度 → 弧度 转换完成');
