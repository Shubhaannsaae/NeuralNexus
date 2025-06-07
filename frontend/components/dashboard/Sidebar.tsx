'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { HomeIcon, FlaskConicalIcon, BoxIcon, NetworkIcon, LightbulbIcon } from 'lucide-react';

export function Sidebar() {
  const pathname = usePathname();

  const navItems = [
    { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
    { name: 'Drug Discovery', href: '/dashboard/discover', icon: FlaskConicalIcon },
    { name: 'Proteins', href: '/dashboard/proteins', icon: BoxIcon },
    { name: 'Knowledge Graph', href: '/dashboard/knowledge-graph', icon: NetworkIcon },
    { name: 'Hypotheses', href: '/dashboard/hypotheses', icon: LightbulbIcon },
  ];

  return (
    <nav className="sidebar">
      <div className="sidebar-header">
        <h2>NeuralNexus</h2>
      </div>
      <ul className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;

          return (
            <li key={item.href}>
              <Link href={item.href} className={`nav-item ${isActive ? 'active' : ''}`}>
                <Icon size={20} />
                <span>{item.name}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
