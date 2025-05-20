
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, BarChart3, Brain, CheckCircle, Cog, Info, XCircle } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line
} from "recharts";

interface ModelOutput {
  fusion_score: number;
  confidence: number;
  image_contribution: number;
  text_contribution: number;
  diagnosis: 'benign' | 'malignant' | 'inconclusive';
  recommendation: string;
  metrics: {
    auc: number;
    f1: number;
    precision: number;
    recall: number;
    specificity: number;
    sensitivity: number;
  },
  prediction_history: Array<{timestamp: string, score: number}>;
}

const metricsData = [
  { name: 'AUC-ROC', model: 0.89, baseline: 0.76 },
  { name: 'F1 Score', model: 0.85, baseline: 0.72 },
  { name: 'Precision', model: 0.82, baseline: 0.68 },
  { name: 'Recall', model: 0.86, baseline: 0.74 },
  { name: 'Specificity', model: 0.88, baseline: 0.71 },
  { name: 'Sensitivity', model: 0.86, baseline: 0.69 }
];

const FusionModel: React.FC<{hasImage: boolean, hasText: boolean}> = ({ hasImage, hasText }) => {
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [modelOutput, setModelOutput] = useState<ModelOutput | null>(null);
  const [progress, setProgress] = useState<number>(0);

  const runFusionModel = () => {
    if (!hasImage || !hasText) return;
    
    setIsRunning(true);
    setProgress(0);
    setModelOutput(null);
    
    // Simulate progress
    const interval = setInterval(() => {
      setProgress(prev => {
        const newProgress = prev + Math.random() * 15;
        if (newProgress >= 100) {
          clearInterval(interval);
          simulateModelOutput();
          return 100;
        }
        return newProgress;
      });
    }, 300);
  };
  
  const simulateModelOutput = () => {
    setTimeout(() => {
      // Generate prediction history with a slightly upward trend
      const predictionHistory = [];
      const baseScore = 0.55 + Math.random() * 0.1; // Base score between 0.55 and 0.65
      
      for (let i = 0; i < 8; i++) {
        const score = baseScore + (i * 0.02) + (Math.random() * 0.04 - 0.02); // Small random variation
        const date = new Date();
        date.setDate(date.getDate() - (7 - i));
        
        predictionHistory.push({
          timestamp: date.toISOString().split('T')[0],
          score: Math.min(0.95, score) // Cap at 0.95
        });
      }
      
      const mockOutput: ModelOutput = {
        fusion_score: predictionHistory[predictionHistory.length - 1].score,
        confidence: 0.75 + Math.random() * 0.2,
        image_contribution: 0.6 + Math.random() * 0.2,
        text_contribution: 0.4 + Math.random() * 0.2,
        diagnosis: predictionHistory[predictionHistory.length - 1].score > 0.7 ? 'malignant' : 'benign',
        recommendation: predictionHistory[predictionHistory.length - 1].score > 0.7 
          ? "Recommend immediate biopsy and surgical consultation." 
          : "Recommend follow-up imaging in 3-6 months to monitor changes.",
        metrics: {
          auc: 0.88 + Math.random() * 0.05,
          f1: 0.84 + Math.random() * 0.05,
          precision: 0.81 + Math.random() * 0.05,
          recall: 0.85 + Math.random() * 0.05,
          specificity: 0.87 + Math.random() * 0.05,
          sensitivity: 0.85 + Math.random() * 0.05
        },
        prediction_history: predictionHistory
      };
      
      setModelOutput(mockOutput);
      setIsRunning(false);
    }, 1500);
  };

  const getDiagnosisBadgeColor = (diagnosis: string) => {
    switch (diagnosis) {
      case 'malignant':
        return "bg-medical-danger text-white";
      case 'benign':
        return "bg-medical-success text-white";
      default:
        return "bg-medical-warning text-white";
    }
  };

  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 0.85) return "High";
    if (confidence >= 0.7) return "Moderate";
    return "Low";
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-medical-primary flex items-center">
          <Brain className="mr-2 h-5 w-5" />
          Multimodal Fusion Analysis
        </CardTitle>
        <CardDescription>Combining image and text analysis for comprehensive assessment</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-3 mb-2">
            <div className="flex items-center gap-1">
              <div className={`h-3 w-3 rounded-full ${hasImage ? 'bg-medical-success' : 'bg-gray-300'}`}></div>
              <span className="text-sm">Image Data</span>
            </div>
            <div className="flex items-center gap-1">
              <div className={`h-3 w-3 rounded-full ${hasText ? 'bg-medical-success' : 'bg-gray-300'}`}></div>
              <span className="text-sm">Text Data</span>
            </div>
          </div>
          
          <Button 
            onClick={runFusionModel} 
            disabled={!hasImage || !hasText || isRunning}
            className="w-full bg-gradient-to-r from-medical-primary to-blue-700 hover:from-medical-primary/90 hover:to-blue-700/90"
          >
            {isRunning ? (
              <>
                <Cog className="mr-2 h-4 w-4 animate-spin" />
                Running Fusion Model...
              </>
            ) : (
              'Run Multimodal Analysis'
            )}
          </Button>
          
          {isRunning && (
            <div className="mt-2">
              <div className="flex justify-between mb-1 text-xs">
                <span>Processing</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          )}

          {modelOutput && (
            <div className="mt-4 space-y-6">
              <div className="p-4 border rounded-lg bg-gray-50">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium">Diagnostic Assessment</h3>
                  <Badge className={getDiagnosisBadgeColor(modelOutput.diagnosis)}>
                    {modelOutput.diagnosis.toUpperCase()}
                  </Badge>
                </div>
                
                <div className="mt-4">
                  <div className="text-sm text-gray-500 mb-1">Malignancy Risk Score</div>
                  <div className="relative pt-1">
                    <div className="overflow-hidden h-4 flex rounded">
                      <div 
                        style={{ width: `${modelOutput.fusion_score * 100}%` }}
                        className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center transition-all ${
                          modelOutput.fusion_score >= 0.7 ? 'bg-medical-danger' : 
                          modelOutput.fusion_score >= 0.4 ? 'bg-medical-warning' : 
                          'bg-medical-success'
                        }`}
                      >
                        <span className="text-xs font-semibold">
                          {Math.round(modelOutput.fusion_score * 100)}%
                        </span>
                      </div>
                    </div>
                    
                    <div className="flex justify-between text-xs mt-1">
                      <span>Low Risk</span>
                      <span>High Risk</span>
                    </div>
                  </div>
                </div>
                
                <div className="mt-4 grid grid-cols-3 gap-3">
                  <div className="bg-white p-3 rounded border">
                    <div className="text-xs text-gray-500">Confidence</div>
                    <div className="flex items-center mt-1">
                      <span className="font-medium">{getConfidenceLevel(modelOutput.confidence)}</span>
                      <span className="text-xs ml-1">({Math.round(modelOutput.confidence * 100)}%)</span>
                    </div>
                  </div>
                  <div className="bg-white p-3 rounded border">
                    <div className="text-xs text-gray-500">Image Contribution</div>
                    <div className="font-medium mt-1">
                      {Math.round(modelOutput.image_contribution * 100)}%
                    </div>
                  </div>
                  <div className="bg-white p-3 rounded border">
                    <div className="text-xs text-gray-500">Text Contribution</div>
                    <div className="font-medium mt-1">
                      {Math.round(modelOutput.text_contribution * 100)}%
                    </div>
                  </div>
                </div>
                
                <div className="mt-4 p-3 bg-white rounded border">
                  <div className="flex gap-2 items-start">
                    {modelOutput.fusion_score >= 0.7 ? (
                      <AlertTriangle className="h-5 w-5 text-medical-danger flex-shrink-0 mt-0.5" />
                    ) : (
                      <Info className="h-5 w-5 text-medical-primary flex-shrink-0 mt-0.5" />
                    )}
                    <div>
                      <div className="font-medium mb-1">Recommendation</div>
                      <p className="text-sm text-gray-600">{modelOutput.recommendation}</p>
                    </div>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="font-medium flex items-center mb-3">
                  <BarChart3 className="h-4 w-4 mr-1" />
                  Prediction Trend
                </h3>
                <div className="h-64 border rounded-lg p-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={modelOutput.prediction_history}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="timestamp" 
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => {
                          const date = new Date(value);
                          return `${date.getMonth() + 1}/${date.getDate()}`;
                        }}
                      />
                      <YAxis 
                        domain={[0, 1]} 
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => `${Math.round(value * 100)}%`}
                      />
                      <Tooltip 
                        formatter={(value) => [`${Math.round(Number(value) * 100)}%`, 'Risk Score']}
                        labelFormatter={(label) => `Date: ${new Date(label).toLocaleDateString()}`}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="score" 
                        stroke="#0EA5E9" 
                        strokeWidth={2}
                        activeDot={{ r: 6 }} 
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              <div>
                <h3 className="font-medium flex items-center mb-3">
                  <CheckCircle className="h-4 w-4 mr-1" />
                  Model Performance Metrics
                </h3>
                <div className="h-64 border rounded-lg p-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={metricsData}
                      margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 1]} tickFormatter={(value) => `${value * 100}%`} />
                      <Tooltip formatter={(value) => [`${Math.round(Number(value) * 100)}%`]} />
                      <Area 
                        type="monotone" 
                        dataKey="model" 
                        stackId="1" 
                        stroke="#0EA5E9" 
                        fill="#E0F2FE" 
                        name="Our Model"
                      />
                      <Area 
                        type="monotone" 
                        dataKey="baseline" 
                        stackId="2" 
                        stroke="#94A3B8" 
                        fill="#F1F5F9" 
                        name="Baseline"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default FusionModel;
