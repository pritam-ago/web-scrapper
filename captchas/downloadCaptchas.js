import axios from 'axios';
import { writeFileSync, mkdirSync, existsSync, readdirSync } from 'fs';
import { join } from 'path';
import { setTimeout as delay } from 'timers/promises';

const CAPTCHA_URL = 'https://sp.srmist.edu.in/srmiststudentportal/captchas'; // 🔁 Update this
const OUTPUT_DIR = join(process.cwd(), 'captchas');
const TOTAL = 500;
const DELAY_MS = 200;

if (!existsSync(OUTPUT_DIR)) {
  mkdirSync(OUTPUT_DIR);
}

const getDownloadedIndexes = () =>
  new Set(readdirSync(OUTPUT_DIR).map(name => parseInt(name.split('.')[0])));

const downloadCaptcha = async (index) => {
  try {
    const response = await axios.get(CAPTCHA_URL, {
      responseType: 'arraybuffer',
      headers: { 'Cache-Control': 'no-cache' },
    });

    const filePath = join(OUTPUT_DIR, `${index}.png`);
    writeFileSync(filePath, response.data);
    console.log(`✅ [${index}] Saved`);
  } catch (err) {
    console.error(`❌ [${index}] Error: ${err.message}`);
  }
};

const main = async () => {
  const downloaded = getDownloadedIndexes();

  for (let i = 1; i <= TOTAL; i++) {
    if (downloaded.has(i)) {
      console.log(`🔁 [${i}] Already exists, skipping`);
      continue;
    }

    await downloadCaptcha(i);
    await delay(DELAY_MS);
  }

  console.log('🎉 Finished downloading remaining CAPTCHAs!');
};

main();
