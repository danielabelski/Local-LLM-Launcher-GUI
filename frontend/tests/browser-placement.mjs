// Built UI acceptance with synthetic hardware/model; no engines or updates start.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
const url = process.argv[2] || 'http://127.0.0.1:18765'
assert.equal(new URL(url).hostname, '127.0.0.1')
const session = `placement-${process.pid}`
function browser(...args) {
  const result = JSON.parse(execFileSync('agent-browser', ['--session', session, '--json', ...args], {encoding:'utf8', timeout:45000}))
  assert.equal(result.success, true, result.error)
  return result.data
}
try {
  browser('open', url)
  console.log(browser('eval', `(async () => {
    const wait = ms => new Promise(r => setTimeout(r, ms));
    const check = (v, message) => { if (!v) throw new Error(message); };
    const original = window.fetch.bind(window);
    let submitted, launches = 0;
    const model = {repo_id:'test/moe', name:'Test MoE', path:'/fake/model.gguf', format:'gguf', size_bytes:1073741824, gguf_files:[]};
    window.fetch = async (path, options) => {
      if (path === '/api/hardware') {
        const hw = await (await original(path, options)).json();
        return Response.json({...hw, numa:{linux:true, node_count:2, nodes:[0,1], numactl_path:'/usr/bin/numactl'},
          engines:{...hw.engines, llamacpp_path:'/fake/llama-server'},
          llama_devices:[0,1,2].map(n=>({name:'CUDA'+n,description:'Synthetic GPU'}))});
      }
      if (path === '/api/models') return Response.json({models:[model]});
      if (path.startsWith('/api/presets')) return Response.json({presets:[{name:'Test',config:{split_mode:'layer'}}]});
      if (path === '/api/advise') {
        submitted = JSON.parse(options.body);
        if (submitted.config.tensor_split === '40,40') return Response.json({detail:'Provide one GPU split proportion per selected device.'},{status:400});
        return Response.json({overall:{level:'yellow',headline:'Memory fit is unknown with custom placement.',details:[]},budget:{fit_unknown:true},flags:{}});
      }
      if (path === '/api/servers' && options?.method === 'POST') { launches++; throw new Error('Never start a real engine in a browser check'); }
      return original(path, options);
    };
    const refresh = [...document.querySelectorAll('button')].find(b=>b.textContent==='Refresh');
    // The first click may share the initial, already-running hardware request.
    // Wait for it, then request the synthetic topology through our interceptor.
    for (let attempt = 0; attempt < 2; attempt++) {
      refresh.click();
      await wait(50);
      const deadline = Date.now() + 10000;
      while (refresh.disabled && Date.now() < deadline) await wait(25);
      check(!refresh.disabled, 'Hardware refresh timed out');
    }
    [...document.querySelectorAll('nav button')].find(b=>b.textContent.endsWith('Launch')).click();
    await wait(700);
    check(document.body.innerText.includes('2 allowed node(s)'), 'Two-node topology missing');
    const advanced = document.querySelector('details');
    if (advanced) advanced.open = true;
    const set = (label, value) => {
      const el = document.querySelector('[aria-label="'+label+'"]');
      check(el, 'Missing '+label);
      const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto,'value').set.call(el,value);
      el.dispatchEvent(new Event(el.tagName==='SELECT'?'change':'input',{bubbles:true}));
    };
    const specs = await (await original('/api/catalog/llamacpp')).json();
    const label = key => specs.flags.find(f=>f.key===key).label;
    for (const name of ['CUDA0','CUDA1','CUDA2']) [...document.querySelectorAll('label')].find(l=>l.textContent===name).querySelector('input').click();
    set(label('tensor_split'),'40,40,40');
    set(label('numa'),'distribute');
    set(label('n_cpu_moe'),'26');
    document.querySelector('[aria-label="'+label('numactl_interleave')+'"]').click();
    await wait(600);
    check(submitted.config.device==='CUDA0,CUDA1,CUDA2', 'Device selection lost');
    check(submitted.config.tensor_split==='40,40,40', 'Ratios lost');
    check(submitted.config.numa==='distribute' && submitted.config.n_cpu_moe===26 && submitted.config.numactl_interleave===true, 'Advanced values lost');
    check(!document.querySelector('.launchbtn').disabled, 'Valid advice cannot launch');
    set(label('tensor_split'),'40,40');
    await wait(600);
    check(document.querySelector('[role="alert"]')?.textContent.includes('one GPU split proportion'), 'Validation error hidden');
    check(document.querySelector('.launchbtn').disabled, 'Invalid config permits launch');
    check(launches===0,'Engine launch attempted');
    return 'Three-card selection, two-node NUMA, expert placement, and validation UI passed';
  })()`).result)
} finally { browser('close') }
