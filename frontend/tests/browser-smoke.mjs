// Run after npm run build, against a local launcher: node tests/browser-smoke.mjs URL
// Settings writes and model/download data are intercepted inside this test session.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'

const url = process.argv[2] || 'http://127.0.0.1:18765'
assert.equal(new URL(url).hostname, '127.0.0.1')
const session = `launcher-smoke-${process.pid}`
function browser(...args) {
  const result = JSON.parse(execFileSync('agent-browser', ['--session', session, '--json', ...args],
    { encoding: 'utf8', timeout: 45000 }))
  assert.equal(result.success, true, result.error)
  return result.data
}
try {
  browser('open', url)
  const result = browser('eval', `
    (async () => {
      const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
      const check = (condition, message) => { if (!condition) throw new Error(message); };
      const clickTab = name => [...document.querySelectorAll('nav button')].find(x => x.textContent.trim().endsWith(name)).click();
      await wait(300);
      check(!document.querySelector('.topbar').innerText.includes('<span>'), 'Breadcrumb markup leaked');
      const original = window.fetch.bind(window);
      let release;
      const held = new Promise(resolve => { release = resolve; });
      let hardwareReads = 0;
      let hardwareStarted = 0;
      let serversStarted = 0;
      window.fetch = async (path, options) => {
        if (path === '/api/hardware') hardwareStarted++;
        if (path === '/api/servers') serversStarted++;
        if (path === '/api/hardware' || path === '/api/servers') await held;
        const response = await original(path, options);
        if (path !== '/api/hardware') return response;
        hardwareReads++;
        const hardware = await response.json();
        return Response.json({...hardware, total_vram_mb: 16384, gpus: [
          {index: 0, name: 'Test GPU', vram_total_mb: 16384, vram_free_mb: 12288, utilization_pct: 0}
        ]});
      };
      const refresh = [...document.querySelectorAll('button')].find(x => x.textContent === 'Refresh');
      try {
        refresh.click();
        await wait(30);
        check(refresh.disabled && refresh.textContent === 'Refreshing…', 'Refresh lacks busy state');
        await wait(6500);
        check(hardwareStarted === 1, 'Hardware polls overlap a pending Refresh request');
        check(serversStarted === 1, 'Server polls overlap a pending Refresh request');
        check(refresh.disabled, 'Refresh completed before its requests finished');
      } finally {
        release();
      }
      await wait(600);
      check(!refresh.disabled, 'Refresh stayed disabled');
      check(document.querySelector('.sb-status').textContent.includes('4.0 / 16 GB'), 'VRAM usage is not total minus free');
      const statusFetch = window.fetch;
      let saved = {hf_token: '********', hf_token_set: true, gguf_folders: [], llamacpp_path: null};
      let submitted;
      let modelReads = 0;
      window.fetch = async (path, options) => {
        if (path === '/api/settings') {
          if (options?.method === 'PUT') {
            submitted = JSON.parse(options.body);
            if ('hf_token' in submitted) saved = {...saved, hf_token_set: !!submitted.hf_token};
          }
          return Response.json(saved);
        }
        if (path === '/api/models') { modelReads++; return Response.json({models: []}); }
        if (path === '/api/downloads') return Response.json({downloads: [{id: 'done-once', status: 'done'}]});
        return statusFetch(path, options);
      };
      clickTab('Settings');
      await wait(200);
      const token = document.querySelector('input[aria-label="Hugging Face access token"]');
      check(token.value === '' && token.placeholder === 'Saved (hidden)', 'Mask populated token input');
      document.querySelector('form').requestSubmit();
      await wait(150);
      check(!('hf_token' in submitted), 'Untouched token was overwritten');
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
      setter.call(token, 'hf_test_replacement'); token.dispatchEvent(new Event('input', {bubbles:true}));
      document.querySelector('form').requestSubmit();
      await wait(150);
      check(submitted.hf_token === 'hf_test_replacement', 'Replacement token was corrupted');
      const remove = [...document.querySelectorAll('label')].find(x => x.textContent.includes('Remove saved token'));
      remove.querySelector('input').click();
      document.querySelector('form').requestSubmit();
      await wait(150);
      check(submitted.hf_token === null, 'Token removal was not explicit');
      clickTab('Models');
      await wait(2600);
      const afterCompletion = modelReads;
      await wait(4300);
      check(modelReads === afterCompletion, 'Completed download repeatedly rescans models');
      check(hardwareReads >= 2, 'Hardware did not poll again');
      return 'Refresh, serialized polling, VRAM, token handling, and download transition passed';
    })()
  `).result
  console.log(result)
} finally {
  browser('close')
}
