'use client';

import React from 'react';
import { SearchIcon, SettingsIcon, UserIcon } from 'lucide-react';

export function Header() {
  return (
    <header className="dashboard-header">
      <div className="header-content">
        <div className="search-bar">
          <SearchIcon size={16} />
          <input 
            type="text" 
            placeholder="Search proteins, compounds, or papers..." 
            className="form-control"
          />
        </div>
        <div className="header-actions">
          <button className="btn btn--outline btn--sm">
            <SettingsIcon size={16} />
          </button>
          <button className="btn btn--outline btn--sm">
            <UserIcon size={16} />
          </button>
        </div>
      </div>
    </header>
  );
}
