// Built UI acceptance with synthetic GPU/runtime evidence; never starts an engine.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
const url = process.argv[2] || 'http://127.0.0.1:18765'
assert.equal(new URL(url).hostname, '127.0.0.1')
const session = `vllm-backends-${process.pid}`
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
    const model = {repo_id:'test/bf16', path:'/fake/model', format:'safetensors', size_gb:1};
    window.fetch = async (path, options) => {
      if (path === '/api/hardware') {
        const hw = await (await original(path, options)).json();
        return Response.json({...hw, gpus:[{index:0,name:'RTX 5060 Ti',compute_capability:'12.0',vram_total_mb:16384,vram_free_mb:15000}],
          engines:{...hw.engines,vllm_native:true,vllm_docker:true}});
      }
      if (path === '/api/models') return Response.json({models:[model]});
      if (path.startsWith('/api/presets')) return Response.json({presets:[{name:'Automatic',config:{}}]});
      if (path === '/api/advise') {
        submitted = JSON.parse(options.body);
        if (submitted.config.attention_backend === 'B12X' && submitted.config.dtype === 'float16')
          return Response.json({detail:'B12X attention requires bfloat16 computation.'},{status:400});
        const message = submitted.engine_mode === 'vllm-docker' ? 'Docker image compatibility is unverified.' : 'Check this backend against your model.';
        return Response.json({overall:{level:'yellow',headline:message,details:[]},budget:{fit_unknown:true},
          flags:{attention_backend:{level:'yellow',message}}});
      }
      if (path === '/api/servers' && options?.method === 'POST') { launches++; throw new Error('Never start an engine in this check'); }
      return original(path, options);
    };
    const refresh = [...document.querySelectorAll('button')].find(b=>b.textContent==='Refresh');
    for (let attempt = 0; attempt < 2; attempt++) {
      refresh.click(); await wait(50);
      const deadline = Date.now()+10000;
      while(refresh.disabled && Date.now()<deadline) await wait(25);
      check(!refresh.disabled,'Refresh timed out');
    }
    [...document.querySelectorAll('nav button')].find(b=>b.textContent.endsWith('Launch')).click();
    await wait(700);
    document.querySelector('details').open = true;
    const specs = await (await original('/api/catalog/vllm')).json();
    const label = key => specs.flags.find(f=>f.key===key).label;
    const set = (key,value) => {
      const el = document.querySelector('[aria-label="'+label(key)+'"]');
      check(el,'Missing '+key);
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(el,value);
      el.dispatchEvent(new Event('change',{bubbles:true}));
    };
    for (const key of ['linear_backend','moe_backend','attention_backend']) {
      check(document.querySelector('[aria-label="'+label(key)+'"]').value==='','Automatic default missing');
    }
    set('linear_backend','flashinfer_cutlass');
    set('moe_backend','b12x');
    set('attention_backend','B12X');
    set('dtype','bfloat16');
    await wait(650);
    check(submitted.config.linear_backend==='flashinfer_cutlass' && submitted.config.moe_backend==='b12x' && submitted.config.attention_backend==='B12X','Selections did not reach API');
    set('dtype','float16'); await wait(650);
    check(document.querySelector('[role="alert"]')?.textContent.includes('bfloat16'),'Incompatibility hidden');
    check(document.querySelector('.launchbtn').disabled,'Invalid config permits launch');
    set('attention_backend',''); set('linear_backend',''); set('moe_backend',''); await wait(650);
    check(!('attention_backend' in submitted.config) && !('linear_backend' in submitted.config) && !('moe_backend' in submitted.config),'Automatic left stale overrides');
    const engine = [...document.querySelectorAll('select')].find(el=>[...el.options].some(o=>o.value==='vllm-docker'));
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(engine,'vllm-docker');
    engine.dispatchEvent(new Event('change',{bubbles:true})); await wait(700);
    document.querySelector('details').open = true;
    set('attention_backend','B12X'); await wait(650);
    check(submitted.engine_mode==='vllm-docker','Docker mode lost');
    check(document.body.innerText.includes('Docker image compatibility is unverified.'),'Container evidence warning hidden');
    check(launches===0,'Engine launch attempted');
    return 'Backend selection, automatic reset, invalid-config blocking, and Docker warning passed';
  })()`).result)
  browser('set', 'viewport', '390', '844')
  const layout = browser('eval', `({width:innerWidth,scroll:document.documentElement.scrollWidth})`).result
  assert.ok(layout.scroll <= layout.width + 1, `Mobile overflow: ${JSON.stringify(layout)}`)
} finally { browser('close') }
