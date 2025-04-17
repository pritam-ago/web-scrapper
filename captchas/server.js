import express from 'express';
import { join } from 'path';
import { writeFileSync, readFileSync, readdirSync, existsSync, appendFileSync } from 'fs';

const app = express();
const PORT = 3000;

const captchaDir = join(process.cwd(), 'captchas');
const labelFile = join(process.cwd(), 'labels.csv');
const publicDir = join(process.cwd(), 'public');

app.use(express.static(publicDir));
app.use(express.json());
// Add this below your other `app.use(...)` lines
app.use('/captchas', express.static(captchaDir));


let files = readdirSync(captchaDir).filter(f => f.endsWith('.png')).sort((a, b) => parseInt(a) - parseInt(b));
let currentIndex = 0;

if (!existsSync(labelFile)) {
  writeFileSync(labelFile, 'filename,label\n');
} else {
  const existing = readFileSync(labelFile, 'utf-8').split('\n').filter(Boolean).slice(1);
  const labeledFiles = new Set(existing.map(l => l.split(',')[0]));
  files = files.filter(f => !labeledFiles.has(f));
}

app.get('/next', (req, res) => {
  if (currentIndex >= files.length) return res.json({ done: true });

  const filename = files[currentIndex++];
  res.json({ filename });
});

app.post('/label', (req, res) => {
  const { filename, label } = req.body;
  if (!filename || !label) return res.status(400).send('Missing fields');

  appendFileSync(labelFile, `${filename},${label}\n`);
  res.sendStatus(200);
});

app.listen(PORT, () => {
  console.log(`🧠 CAPTCHA labeler running at http://localhost:${PORT}`);
});
