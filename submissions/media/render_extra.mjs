import { chromium } from 'playwright-core';
import path from 'node:path';
const here = path.dirname(new URL(import.meta.url).pathname);
const b = await chromium.launch({ executablePath: process.env.DEMO_CHROME });
const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
await p.goto('file://' + path.join(here, 'video_extra.html'), { waitUntil: 'networkidle' });
await p.evaluate(() => document.fonts.ready);
await p.locator('section.slide').screenshot({ path: path.join(here, '../build/slide-midlap.png') });
await b.close();
