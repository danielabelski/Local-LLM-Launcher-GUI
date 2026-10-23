// Run against the built launcher or Vite: node tests/browser-updater.mjs URL
// All settings and updater requests are intercepted; no source build or save occurs.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'

const url = process.argv[2] || 'http://127.0.0.1:18765'
assert.equal(new URL(url).hostname, '127.0.0.1')
const session = `launcher-updater-${process.pid}`
function browser(...args) {
  const result = JSON.parse(execFileSync('agent-browser', ['--session', session, '--json', ...args],
    { encoding: 'utf8', timeout: 45000 }))
  assert.equal(result.success, true, result.error)
  return result.data
}
try {
  browser('open', url)
  console.log(browser('eval', `
    (async () => {
      const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
      const check = (ok, message) => { if (!ok) throw new Error(message); };
      const until = async (condition, message) => {
        for (let i = 0; i < 150; i++) { if (condition()) return; await wait(40); }
        throw new Error(message);
      };
      const clickTab = name => [...document.querySelectorAll('nav button')].find(x => x.textContent.trim().endsWith(name)).click();
      const button = text => [...document.querySelectorAll('button')].find(x => x.textContent === text);
      const pathInput = () => document.querySelector('input[placeholder="/path/to/llama-server"]');
      const setInput = (input, value) => {
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, value);
        input.dispatchEvent(new Event('input', {bubbles:true}));
      };
      await until(() => document.querySelector('nav button'), 'App did not mount');
      const original = window.fetch.bind(window);
      let settings = {gguf_folders: [], llamacpp_path: '/old/llama-server', vllm_path: null};
      let status = {state: 'idle'};
      const heldReads = [];
      const releaseIdle = () => { holdIdle = false; heldReads.forEach(resolve => resolve()); };
      let holdIdle = true;
      let statusRequests = 0;
      let buildNumber = 0;
      let submitted;
      window.fetch = async (path, options) => {
        if (path === '/api/settings') {
          if (options?.method === 'PUT') { submitted = JSON.parse(options.body); settings = {...settings, ...submitted}; }
          return Response.json(settings);
        }
        if (path.startsWith('/api/updates/check/')) return Response.json({
          engine: 'llamacpp', backend: 'cpu', jobs: 2, reasons: [], supported: true,
          check_id: 'test-check', revision: 'test-revision', current_path: settings.llamacpp_path
        });
        if (path === '/api/updates') {
          if (options?.method === 'POST') {
            buildNumber++;
            status = {state: 'running', engine: 'llamacpp', path: '/build/' + buildNumber, stage: 'Compiling test source'};
            return Response.json(status);
          }
          statusRequests++;
          if (holdIdle) {
            await new Promise(resolve => { heldReads.push(resolve); });
            return Response.json({state: 'idle'});
          }
          return Response.json(status);
        }
        return original(path, options);
      };
      clickTab('Settings');
      await until(() => pathInput()?.value === '/old/llama-server' && heldReads.length, 'Settings did not load');
      // Development React may mount effects twice; neither active poll may overlap itself.
      const initialReads = statusRequests;
      await wait(2200);
      check(statusRequests === initialReads, 'Updater status polls overlap');
      button('Check requirements and source').click();
      await until(() => button('Build and use this revision'), 'Build check did not finish');
      button('Build and use this revision').click();
      await until(() => document.body.textContent.includes('llamacpp: running'), 'Start did not show running');
      releaseIdle();
      await wait(200);
      check(document.body.textContent.includes('llamacpp: running'), 'Delayed idle poll erased running build');
      check(button('Save settings').disabled, 'Stale status re-enabled settings save during build');
      status = {...status, state: 'complete', executable: '/managed/llama-server', stage: 'Ready'};
      settings.llamacpp_path = status.executable;
      await until(() => pathInput().value === status.executable, 'Successful build did not update executable field');
      setInput(pathInput(), '/manual/rollback');
      await wait(40);
      document.querySelector('form').requestSubmit();
      await until(() => submitted?.llamacpp_path === '/manual/rollback', 'Manual rollback was not saved');
      clickTab('Models');
      await wait(100);
      clickTab('Settings');
      await until(() => pathInput()?.value === '/manual/rollback', 'Saved rollback was replaced by old completion on remount');
      await wait(2200);
      check(pathInput().value === '/manual/rollback', 'Completed job replay undid rollback');
      button('Check requirements and source').click();
      await until(() => button('Build and use this revision'), 'Second check failed');
      button('Build and use this revision').click();
      await until(() => document.body.textContent.includes('llamacpp: running'), 'Second start failed');
      setInput(pathInput(), '/manual/newer-choice');
      await wait(40);
      status = {...status, state: 'complete', executable: '/managed/second', stage: 'Ready'};
      await until(() => document.body.textContent.includes('llamacpp: complete'), 'Second completion missing');
      check(pathInput().value === '/manual/newer-choice', 'Completion overwrote a newer manual path edit');
      button('Build and use this revision').click();
      await until(() => document.body.textContent.includes('llamacpp: running'), 'Third start failed');
      status = {...status, state: 'failed', error: 'Mock compiler failure', stage: 'Failed', log: 'Mock build log retained'};
      await until(() => document.body.textContent.includes('Mock compiler failure'), 'Build failure missing');
      check(document.querySelector('.logbox')?.textContent.includes('Mock build log retained'), 'Failure log missing');
      check(!button('Save settings').disabled, 'Failed build left save disabled');
      check(pathInput().value === '/manual/newer-choice', 'Failed build changed executable');
      return 'Updater polling order, completion, manual rollback/remount, newer edits, and failure display passed';
    })()
  `).result)
} finally {
  browser('close')
}
