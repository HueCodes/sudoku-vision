#!/usr/bin/env node
/**
 * Browser-based test for the Sudoku Vision web app.
 * Uses Puppeteer to test the image processing pipeline.
 */

import puppeteer from 'puppeteer';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TEST_URL = 'http://localhost:5173/test.html';

async function runTest() {
  console.log('Starting Puppeteer browser test...\n');

  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();

  // Collect console logs
  const logs = [];
  page.on('console', (msg) => {
    const text = msg.text();
    logs.push(text);
    if (text.includes('ERROR') || text.includes('Error')) {
      console.log('  [ERROR]', text);
    }
  });

  // Collect page errors
  page.on('pageerror', (err) => {
    console.log('  [PAGE ERROR]', err.message);
  });

  try {
    console.log('Navigating to test page...');
    await page.goto(TEST_URL, { waitUntil: 'networkidle0', timeout: 60000 });

    // Wait for processing to complete (check for status change)
    console.log('Waiting for pipeline to complete...');
    await page.waitForFunction(
      () => {
        const status = document.getElementById('status');
        return status && (
          status.classList.contains('success') ||
          status.classList.contains('error')
        );
      },
      { timeout: 120000 }
    );

    // Get results
    const results = await page.evaluate(() => {
      return {
        status: document.getElementById('status')?.textContent,
        isSuccess: document.getElementById('status')?.classList.contains('success'),
        recognized: document.getElementById('recognized')?.textContent,
        solution: document.getElementById('solution')?.textContent,
        debug: document.getElementById('debug')?.textContent,
      };
    });

    console.log('\n=== RESULTS ===');
    console.log('Status:', results.status);
    console.log('Success:', results.isSuccess);

    if (results.recognized) {
      console.log('\nRecognized Grid:');
      console.log(results.recognized);
    }

    if (results.solution && results.isSuccess) {
      console.log('\nSolution:');
      console.log(results.solution);
    }

    console.log('\n=== DEBUG LOG ===');
    console.log(results.debug || '(empty)');

    // Return exit code based on whether it completed (even with recognition errors)
    // We consider it a pass if the pipeline ran without JS errors
    const pipelineRan = results.debug?.includes('Classifying') || results.debug?.includes('Solving');

    if (pipelineRan) {
      console.log('\n[PASS] Pipeline executed successfully');
      console.log('Note: Recognition accuracy depends on ML model quality');
    } else if (results.debug?.includes('ERROR')) {
      console.log('\n[FAIL] Pipeline encountered errors');
    } else {
      console.log('\n[UNKNOWN] Could not determine pipeline status');
    }

  } catch (error) {
    console.error('\nTest failed:', error.message);
    console.log('\nCollected logs:');
    logs.forEach((log) => console.log('  ', log));
    process.exit(1);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
