"use client";

import SearchModule from "@/components/SearchModule";

export default function Home() {
  return (
    <main className="min-h-screen bg-[#0f172a] text-white flex flex-col items-center justify-center p-4">
      {/* SearchModule이 모든 기능(검색, 3D 뷰어, 차트, 컨트롤러)을 
        포함하고 있으므로, 여기서는 이것만 불러오면 됩니다. 
      */}
      <SearchModule />
    </main>
  );
}