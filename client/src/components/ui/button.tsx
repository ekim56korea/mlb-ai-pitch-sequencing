import * as React from "react"
// ❌ 삭제된 부분: import { cn } from "@/lib/utils" 또는 "../../lib/utils"

// ✅ 추가된 부분: 라이브러리에서 직접 가져와서 함수를 여기서 정의
import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

// cn 함수를 이 파일 안에서 직접 만듭니다
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline";
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none disabled:opacity-50 h-9 px-4 py-2",
          variant === "default" ? "bg-mlb-navy text-white hover:bg-slate-800" : "border border-slate-200 bg-white hover:bg-slate-100 text-slate-900",
          className
        )}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"
export { Button }