const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
async function post(path: string, body: any){
  const res = await fetch(`${BASE}${path}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
  if(!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}
export function scoreLive(payload:{symbol:string,country:string,query:string}){ return post('/score-live', payload) }
export function scoreManual(payload:{features:Record<string,number>, symbol?:string}){ return post('/score-manual', payload) }
