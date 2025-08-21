import { useState } from 'react'
import { scoreLive, scoreManual } from './api/client'

export default function App(){
  const [symbol, setSymbol] = useState('RELIANCE.NS')
  const [country, setCountry] = useState('IN')
  const [query, setQuery] = useState('Reliance Industries')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const runLive = async () => {
    setLoading(true)
    try{
      const r = await scoreLive({symbol, country, query})
      setResult(r)
    } catch(e:any){
      alert(e.message)
    } finally { setLoading(false) }
  }

  const runManual = async () => {
    setLoading(true)
    try{
      const r = await scoreManual({features: {
        ret_30d: 0.05, vol_30d: 0.02, momentum_7d: 0.012,
        gdp_growth: 6.1, inflation: 5.2, news_sentiment: 0.1, event_risk: -0.2
      }, symbol: 'MANUAL'})
      setResult(r)
    } catch(e:any){
      alert(e.message)
    } finally { setLoading(false) }
  }

  return (
    <div className="container">
      <h1>CredXplain <span className="badge">v2</span></h1>
      <p className="muted">Explainable credit scoring with live data + event extraction, trends & alerts.</p>

      <div className="card">
        <h2>Live Scoring</h2>
        <div className="grid">
          <div><label>Ticker</label><br/><input value={symbol} onChange={e=>setSymbol(e.target.value)} /></div>
          <div><label>Country</label><br/><input value={country} onChange={e=>setCountry(e.target.value)} /></div>
          <div style={{gridColumn:'1 / span 2'}}><label>News Query</label><br/><input value={query} onChange={e=>setQuery(e.target.value)} /></div>
        </div>
        <div style={{marginTop:12}}>
          <button onClick={runLive} disabled={loading}>{loading ? 'Scoring…' : 'Score from Live Data'}</button>
          <button onClick={runManual} disabled={loading} style={{marginLeft:10, background:'#64748b'}}>Manual Example</button>
        </div>
      </div>

      {result && (
        <div className="card">
          <h2>Result</h2>
          <div style={{fontSize:22,fontWeight:700}}>Score: {result.score}</div>
          <div className="muted">P(good): {result.probability_good}</div>
          <p style={{marginTop:10}}>{result.explanation}</p>

          {result.alerts?.length ? (
            <div style={{marginTop:8}}>
              {result.alerts.map((a:any, idx:number)=> (<span key={idx} className="alert">{a.message}</span>))}
            </div>
          ) : null}

          <h3 style={{marginTop:18}}>Top Contributions</h3>
          <pre>{JSON.stringify(result.top_contributions, null, 2)}</pre>

          <h3>Trends</h3>
          <pre>{JSON.stringify(result.trends, null, 2)}</pre>

          <h3>Detected Events</h3>
          {result.events?.length ? (
            <ul>
              {result.events.map((ev:any, i:number)=> (
                <li key={i} style={{marginBottom:6}}>
                  <span className="tag">{ev.impact > 0 ? 'positive' : 'negative'}</span>
                  {ev.headline}
                </li>
              ))}
            </ul>
          ) : (<p className="muted">No key events detected in recent headlines.</p>)}

          <h3>Features Used</h3>
          <pre>{JSON.stringify(result.features_used, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}
