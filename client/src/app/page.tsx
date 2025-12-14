import SearchModule from "../components/SearchModule";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-[#0f172a] to-black">
      <div className="w-full max-w-5xl space-y-12 text-center">
        
        {/* 타이틀 섹션 */}
        <div className="space-y-6 animate-in fade-in slide-in-from-top-4 duration-1000">
          <div className="inline-flex items-center px-4 py-2 rounded-full border border-blue-500/30 bg-blue-500/10 text-blue-400 text-sm font-medium mb-4">
            <span className="relative flex h-2 w-2 mr-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
            </span>
            System Online: v7.0 Zero-Cost Edition
          </div>
          
          <h1 className="text-6xl md:text-8xl font-bold text-white tracking-tighter">
            Pitch <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">Commander</span>
          </h1>
          
          <p className="text-slate-400 text-xl md:text-2xl max-w-2xl mx-auto font-light leading-relaxed">
            Advanced Physics & AI Analytics for MLB<br/>
            <span className="text-sm text-slate-500">Powered by Next.js + FastAPI</span>
          </p>
        </div>

        {/* 우리가 만든 검색 모듈 */}
        <SearchModule />

      </div>
    </main>
  );
}