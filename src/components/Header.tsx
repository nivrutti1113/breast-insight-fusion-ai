
import React from 'react';
import { Brain, FileText, GitMerge, ShieldAlert } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="w-full bg-white shadow-sm mb-6">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <ShieldAlert className="h-8 w-8 text-medical-primary" />
            <div className="ml-3">
              <h1 className="text-2xl font-bold text-gray-800">MediVision AI</h1>
              <p className="text-sm text-gray-500">Breast Cancer Early Detection System</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-6 text-gray-500">
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              <span className="text-sm font-medium">Image Model</span>
            </div>
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              <span className="text-sm font-medium">Text Model</span>
            </div>
            <div className="flex items-center gap-2">
              <GitMerge className="h-5 w-5" />
              <span className="text-sm font-medium">Fusion</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
