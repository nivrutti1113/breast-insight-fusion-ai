
import React from 'react';
import { Github, Mail, Shield } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="w-full bg-white shadow-sm mt-10 py-6">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0 flex items-center">
            <Shield className="h-5 w-5 text-medical-primary mr-2" />
            <span className="text-sm text-gray-600">
              MediVision AI © {new Date().getFullYear()} - Demo System for Research Purposes Only
            </span>
          </div>
          
          <div className="flex items-center space-x-4">
            <a 
              href="#" 
              className="text-gray-500 hover:text-medical-primary transition-colors"
              aria-label="GitHub"
            >
              <Github className="h-5 w-5" />
            </a>
            <a 
              href="mailto:contact@example.com" 
              className="text-gray-500 hover:text-medical-primary transition-colors"
              aria-label="Email"
            >
              <Mail className="h-5 w-5" />
            </a>
          </div>
        </div>
        
        <div className="mt-4 text-xs text-center text-gray-400">
          This system is intended for demonstration and research purposes only. Not for clinical use or diagnosis.
        </div>
      </div>
    </footer>
  );
};

export default Footer;
