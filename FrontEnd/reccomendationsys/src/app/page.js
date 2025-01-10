"use client";

import { useState } from "react";

export default function Home() {
  const [input1, setInput1] = useState("");
  const [input2, setInput2] = useState("");

  return (
    <div className="min-h-screen p-8 pb-20 sm:p-20 font-[family-name:var(--font-geist-sans)] bg-black relative">
      <div className="absolute bottom-0 left-0 w-full h-1/2 bg-gradient-to-t from-blue-500 to-transparent opacity-50 pointer-events-none"></div>
      <main className="flex items-center justify-center min-h-screen">
        <div className="flex gap-4 items-center">
          <select
            value={input1}
            onChange={(e) => setInput1(e.target.value)}
            className="rounded-lg p-2 text-black"
          >
            <option value="" disabled>Select an option</option>
            <option value="Option 1">Option 1</option>
            <option value="Option 2">Option 2</option>
            <option value="Option 3">Option 3</option>
          </select>
          <select
            value={input2}
            onChange={(e) => setInput2(e.target.value)}
            className="rounded-lg p-2 text-black"
          >
            <option value="" disabled>Select an option</option>
            <option value="Option 1">Option 1</option>
            <option value="Option 2">Option 2</option>
            <option value="Option 3">Option 3</option>
          </select>
        </div>
        <div className="bg-white text-black p-4 rounded-lg shadow-lg ml-8">
          <h2 className="text-lg font-semibold">Output</h2>
          <p>Input 1: {input1}</p>
          <p>Input 2: {input2}</p>
        </div>
      </main>
    </div>
  );
}