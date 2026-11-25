"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "מניות" },
  { href: "/expenses", label: "הוצאות Max" },
];

export default function Nav() {
  const pathname = usePathname();
  return (
    <nav className="topnav">
      <div className="brand">
        <span className="brand-mark" />
        <span>Finance Hub</span>
      </div>
      <div className="nav-links">
        {LINKS.map((l) => {
          const active = pathname === l.href;
          return (
            <Link key={l.href} href={l.href} className={active ? "nav-link active" : "nav-link"}>
              {l.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
