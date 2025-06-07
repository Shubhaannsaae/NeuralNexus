'use client';

import React from 'react';
import Link from 'next/link';
import {
  FlaskConicalIcon,
  BoxIcon,
  NetworkIcon,
  LightbulbIcon,
  SearchIcon,
  SettingsIcon,
  UserIcon,
} from 'lucide-react';

export default function LandingPage() {
  const features = [
    {
      icon: <FlaskConicalIcon size={32} />,
      title: 'Drug Discovery',
      description:
        'Advanced compound analysis with ADMET prediction and molecular property optimization.',
    },
    {
      icon: <BoxIcon size={32} />,
      title: 'Protein Analysis',
      description:
        '3D structure visualization and binding site analysis for target identification.',
    },
    {
      icon: <NetworkIcon size={32} />,
      title: 'Knowledge Graph',
      description: 'Interactive exploration of biomedical relationships and pathway connections.',
    },
    {
      icon: <LightbulbIcon size={32} />,
      title: 'AI Hypotheses',
      description: 'Generate novel research hypotheses using advanced machine learning algorithms.',
    },
  ];

  const stats = [
    { value: '50,000+', label: 'Proteins Analyzed' },
    { value: '2.1M+', label: 'Drug Compounds' },
    { value: '500K+', label: 'Research Papers' },
    { value: '10,000+', label: 'Hypotheses Generated' },
  ];

  return (
    <div id="landing-page" className="page active">
      {/* Navigation */}
      <nav className="landing-nav">
        <div className="container flex items-center justify-between py-16">
          <div className="flex items-center gap-8">
            <div className="logo">
              <h2 className="text-2xl font-bold text-white">NeuralNexus</h2>
            </div>
          </div>
          <div className="flex items-center gap-16">
            <Link href="/dashboard" className="btn btn--outline">
              Dashboard
            </Link>
            <Link href="/dashboard" className="btn btn--primary">
              Get Started
            </Link>
          </div>
        </div>
      </nav>

      <main>
        {/* Hero Section */}
        <section className="hero">
          <div className="container">
            <div className="hero-content">
              <h1>Accelerate Neurological Drug Discovery</h1>
              <p>
                Harness the power of AI to discover breakthrough treatments for neurological
                disorders. Our platform combines protein analysis, knowledge graphs, and hypothesis
                generation.
              </p>
              <div className="hero-actions">
                <Link href="/dashboard" className="btn btn--primary btn--lg">
                  Start Discovering
                </Link>
                <button className="btn btn--outline btn--lg">Learn More</button>
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="features">
          <div className="container">
            <h2>Powerful Features for Drug Discovery</h2>
            <div className="features-grid">
              {features.map((feature, index) => (
                <div key={index} className="feature-card">
                  <div className="feature-icon">{feature.icon}</div>
                  <h3>{feature.title}</h3>
                  <p>{feature.description}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Stats Section */}
        <section className="stats">
          <div className="container">
            <div className="stats-grid">
              {stats.map((stat, index) => (
                <div key={index} className="stat-item">
                  <h3>{stat.value}</h3>
                  <p>{stat.label}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
