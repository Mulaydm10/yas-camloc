// slides.html -> submissions/slides.pdf (6 pages, 1920x1080) + build/slide-N.png for the video
import { chromium } from 'playwright-core';
import path from 'node:path';
const here = path.dirname(new URL(import.meta.url).pathname);
const exe = process.env.DEMO_CHROME;
const b = await chromium.launch(exe ? { executablePath: exe } : {});
const p = await b.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await p.goto('file://' + path.join(here, 'slides.html'), { waitUntil: 'networkidle' });
await p.evaluate(() => document.fonts.ready);
const n = await p.locator('section.slide').count();
for (let i = 0; i < n; i++) {
  await p.locator('section.slide').nth(i).screenshot({ path: path.join(here, '../build', `slide-${i + 1}.png`) });
}
await p.pdf({ path: path.join(here, '../slides.pdf'), width: '1920px', height: '1080px', printBackground: true, preferCSSPageSize: true });
await b.close();
console.log('slides', n);
