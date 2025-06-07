'use client';

import React from 'react';
import { FolderIcon, TargetIcon, BeakerIcon, TrendingUpIcon } from 'lucide-react';

export function StatsCards() {
  const stats = [
    {
      title: 'Active Projects',
      value: '12',
      icon: <FolderIcon size={24} />
    },
    {
      title: 'Protein Targets', 
      value: '847',
      icon: <TargetIcon size={24} />
    },
    {
      title: 'Compounds Screened',
      value: '15.2K', 
      icon: <BeakerIcon size={24} />
    },
    {
      title: 'Success Rate',
      value: '78%',
      icon: <TrendingUpIcon size={24} />
    }
  ];

  return (
    <div className="stats-cards">
      {stats.map((stat, index) => (
        <div key={index} className="card stats-card">
          <div className="card__body">
            <div className="stat-info">
              <h3>{stat.title}</h3>
              <div className="stat-value">{stat.value}</div>
            </div>
            <div className="stat-icon">
              {stat.icon}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
