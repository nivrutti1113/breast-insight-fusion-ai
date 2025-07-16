
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, Brain, FileText, Home } from 'lucide-react';
import { Button } from './ui/button';
import { ModeToggle } from './ui/mode-toggle';

export const Header: React.FC = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Home', icon: Home },
    { path: '/analysis', label: 'Analysis', icon: Brain },
    { path: '/results', label: 'Results', icon: FileText },
  ];

  return (
    <header className="bg-white shadow-sm border-b">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Activity className="h-8 w-8 text-blue-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Breast Cancer AI
              </h1>
              <p className="text-sm text-gray-600">
                Deep Learning Mammogram Analysis
              </p>
            </div>
          </div>

          <nav className="flex items-center space-x-4">
            {navItems.map(({ path, label, icon: Icon }) => (
              <Button
                key={path}
                variant={location.pathname === path ? 'default' : 'ghost'}
                size="sm"
                asChild
              >
                <Link to={path} className="flex items-center space-x-2">
                  <Icon className="h-4 w-4" />
                  <span>{label}</span>
                </Link>
              </Button>
            ))}
            <ModeToggle />
          </nav>
        </div>
      </div>
    </header>
  );
};
