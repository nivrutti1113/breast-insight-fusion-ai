import React from 'react';
import { Link } from 'react-router-dom';
import { Brain, FileText, Shield, Upload, Zap } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';

export const HomePage: React.FC = () => {
  const features = [
    {
      icon: Brain,
      title: 'Deep Learning Analysis',
      description: 'Advanced CNN with ResNet50 architecture trained on mammography images',
      color: 'bg-blue-100 text-blue-600'
    },
    {
      icon: FileText,
      title: 'Comprehensive Reports',
      description: 'Detailed PDF reports with analysis results and recommendations',
      color: 'bg-green-100 text-green-600'
    },
    {
      icon: Zap,
      title: 'Grad-CAM Visualization',
      description: 'Visual heatmaps showing model attention areas for interpretability',
      color: 'bg-purple-100 text-purple-600'
    },
    {
      icon: Shield,
      title: 'Medical Grade Security',
      description: 'Secure processing with privacy-first approach for sensitive medical data',
      color: 'bg-red-100 text-red-600'
    }
  ];

  const stats = [
    { label: 'Accuracy', value: '94.5%' },
    { label: 'Sensitivity', value: '92.1%' },
    { label: 'Specificity', value: '96.3%' },
    { label: 'Processing Time', value: '<5s' }
  ];

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="text-center space-y-6">
        <div className="space-y-4">
          <Badge variant="secondary" className="px-4 py-2">
            AI-Powered Medical Analysis
          </Badge>
          <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Breast Cancer Detection
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Advanced deep learning system for mammogram analysis with explainable AI 
            and comprehensive reporting for medical professionals.
          </p>
        </div>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button size="lg" asChild className="text-lg px-8">
            <Link to="/analysis">
              <Upload className="mr-2 h-5 w-5" />
              Start Analysis
            </Link>
          </Button>
          <Button size="lg" variant="outline" asChild className="text-lg px-8">
            <Link to="/results">
              <FileText className="mr-2 h-5 w-5" />
              View Results
            </Link>
          </Button>
        </div>
      </section>

      {/* Stats Section */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <Card key={index} className="text-center">
            <CardHeader className="pb-2">
              <CardTitle className="text-2xl text-blue-600">{stat.value}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">{stat.label}</p>
            </CardContent>
          </Card>
        ))}
      </section>

      {/* Features Section */}
      <section className="space-y-8">
        <div className="text-center space-y-2">
          <h2 className="text-3xl font-bold text-gray-900">Key Features</h2>
          <p className="text-gray-600">
            Advanced AI capabilities for accurate breast cancer detection
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Card key={index} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className={`w-12 h-12 rounded-lg ${feature.color} flex items-center justify-center mb-4`}>
                    <Icon className="h-6 w-6" />
                  </div>
                  <CardTitle className="text-xl">{feature.title}</CardTitle>
                  <CardDescription className="text-gray-600">
                    {feature.description}
                  </CardDescription>
                </CardHeader>
              </Card>
            );
          })}
        </div>
      </section>

      {/* How It Works Section */}
      <section className="space-y-8">
        <div className="text-center space-y-2">
          <h2 className="text-3xl font-bold text-gray-900">How It Works</h2>
          <p className="text-gray-600">
            Simple three-step process for mammogram analysis
          </p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-6">
          <Card className="text-center">
            <CardHeader>
              <div className="w-12 h-12 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-xl font-bold">1</span>
              </div>
              <CardTitle>Upload Image</CardTitle>
              <CardDescription>
                Upload your mammogram image in JPEG, PNG, or DICOM format
              </CardDescription>
            </CardHeader>
          </Card>
          
          <Card className="text-center">
            <CardHeader>
              <div className="w-12 h-12 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-xl font-bold">2</span>
              </div>
              <CardTitle>AI Analysis</CardTitle>
              <CardDescription>
                Our deep learning model analyzes the image for potential abnormalities
              </CardDescription>
            </CardHeader>
          </Card>
          
          <Card className="text-center">
            <CardHeader>
              <div className="w-12 h-12 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-xl font-bold">3</span>
              </div>
              <CardTitle>Get Report</CardTitle>
              <CardDescription>
                Receive comprehensive PDF report with analysis and recommendations
              </CardDescription>
            </CardHeader>
          </Card>
        </div>
      </section>

      {/* Disclaimer Section */}
      <section className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
        <div className="flex items-start space-x-3">
          <Shield className="h-6 w-6 text-yellow-600 mt-1" />
          <div>
            <h3 className="font-semibold text-yellow-800">Important Medical Disclaimer</h3>
            <p className="text-yellow-700 mt-1">
              This AI system is designed for screening purposes and should not replace professional medical diagnosis. 
              Always consult with qualified healthcare professionals for medical decisions. 
              The system is intended to assist healthcare providers in their diagnostic process.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};