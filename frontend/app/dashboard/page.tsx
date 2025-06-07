'use client';

import React from 'react';
import { StatsCards } from '@/components/dashboard/StatsCards';

export default function DashboardHome() {
  return (
    <div className="dashboard-page active">
      <div className="page-header">
        <h1>Dashboard</h1>
        <p>Welcome to NeuralNexus - your comprehensive platform for neurological drug discovery</p>
      </div>

      <StatsCards />

      <div className="dashboard-grid">
        <div className="card">
          <div className="card__body">
            <h3>Recent Activity</h3>
            <div className="activity-list">
              <div className="activity-item">
                <span className="activity-time">2 hours ago</span>
                <span>Protein P04637 analysis completed</span>
              </div>
              <div className="activity-item">
                <span className="activity-time">5 hours ago</span>
                <span>New hypothesis generated for Alzheimer's pathway</span>
              </div>
              <div className="activity-item">
                <span className="activity-time">1 day ago</span>
                <span>Knowledge graph updated with 1,247 new connections</span>
              </div>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="card__body">
            <h3>Discovery Progress</h3>
            <div
              style={{
                height: '200px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: '#f8fafc',
                borderRadius: '8px',
              }}
            >
              <p className="text-secondary">Chart visualization would be displayed here</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
