import React, { useState, useEffect } from 'react';
import { FileText, Download, Eye, Calendar, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Alert, AlertDescription } from '../components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';

interface AnalysisHistory {
  id: string;
  filename: string;
  timestamp: string;
  classification: string;
  probability: number;
  confidence: number;
  reportPath?: string;
  heatmapPath?: string;
}

export const ResultsPage: React.FC = () => {
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisHistory[]>([]);
  const [selectedAnalysis, setSelectedAnalysis] = useState<AnalysisHistory | null>(null);

  useEffect(() => {
    // Load analysis history from localStorage
    const savedHistory = localStorage.getItem('analysisHistory');
    if (savedHistory) {
      setAnalysisHistory(JSON.parse(savedHistory));
    }
  }, []);

  const getClassificationBadge = (classification: string) => {
    const variant = classification === 'Malignant' ? 'destructive' : 'secondary';
    const icon = classification === 'Malignant' ? AlertCircle : CheckCircle;
    const Icon = icon;
    
    return (
      <Badge variant={variant} className="flex items-center space-x-1">
        <Icon className="h-3 w-3" />
        <span>{classification}</span>
      </Badge>
    );
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return 'text-green-600';
    if (confidence > 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const exportToCSV = () => {
    const csvContent = [
      ['Filename', 'Timestamp', 'Classification', 'Probability', 'Confidence'],
      ...analysisHistory.map(analysis => [
        analysis.filename,
        analysis.timestamp,
        analysis.classification,
        analysis.probability.toFixed(3),
        analysis.confidence.toFixed(3)
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis_history_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };

  const clearHistory = () => {
    setAnalysisHistory([]);
    setSelectedAnalysis(null);
    localStorage.removeItem('analysisHistory');
  };

  const getSummaryStats = () => {
    if (analysisHistory.length === 0) return null;

    const malignantCount = analysisHistory.filter(a => a.classification === 'Malignant').length;
    const benignCount = analysisHistory.filter(a => a.classification === 'Benign').length;
    const avgConfidence = analysisHistory.reduce((sum, a) => sum + a.confidence, 0) / analysisHistory.length;

    return {
      total: analysisHistory.length,
      malignant: malignantCount,
      benign: benignCount,
      avgConfidence: avgConfidence
    };
  };

  const stats = getSummaryStats();

  return (
    <div className="space-y-8">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold text-gray-900">Analysis Results</h1>
        <p className="text-gray-600">
          View and manage your mammogram analysis history
        </p>
      </div>

      {analysisHistory.length === 0 ? (
        <Card>
          <CardContent className="text-center py-12">
            <FileText className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">No Analysis History</h3>
            <p className="text-gray-600 mb-4">
              You haven't performed any mammogram analyses yet.
            </p>
            <Button asChild>
              <a href="/analysis">Start Analysis</a>
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-8">
          {/* Summary Statistics */}
          {stats && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-2xl text-blue-600">{stats.total}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">Total Analyses</p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-2xl text-red-600">{stats.malignant}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">Malignant</p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-2xl text-green-600">{stats.benign}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">Benign</p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-2xl text-purple-600">
                    {(stats.avgConfidence * 100).toFixed(1)}%
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">Avg Confidence</p>
                </CardContent>
              </Card>
            </div>
          )}

          {/* Controls */}
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold">Analysis History</h2>
            <div className="space-x-2">
              <Button variant="outline" onClick={exportToCSV}>
                <Download className="mr-2 h-4 w-4" />
                Export CSV
              </Button>
              <Button variant="outline" onClick={clearHistory}>
                Clear History
              </Button>
            </div>
          </div>

          <Tabs defaultValue="grid" className="w-full">
            <TabsList>
              <TabsTrigger value="grid">Grid View</TabsTrigger>
              <TabsTrigger value="list">List View</TabsTrigger>
            </TabsList>
            
            <TabsContent value="grid">
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {analysisHistory.map((analysis) => (
                  <Card 
                    key={analysis.id} 
                    className="cursor-pointer hover:shadow-md transition-shadow"
                    onClick={() => setSelectedAnalysis(analysis)}
                  >
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <div className="space-y-1">
                          <CardTitle className="text-sm font-medium truncate">
                            {analysis.filename}
                          </CardTitle>
                          <CardDescription className="flex items-center space-x-1">
                            <Calendar className="h-3 w-3" />
                            <span className="text-xs">
                              {formatTimestamp(analysis.timestamp)}
                            </span>
                          </CardDescription>
                        </div>
                        {getClassificationBadge(analysis.classification)}
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Probability:</span>
                          <span className="font-mono">
                            {(analysis.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Confidence:</span>
                          <span className={`font-mono ${getConfidenceColor(analysis.confidence)}`}>
                            {(analysis.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>
            
            <TabsContent value="list">
              <div className="space-y-4">
                {analysisHistory.map((analysis) => (
                  <Card key={analysis.id}>
                    <CardContent className="py-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <div className="flex-1">
                            <h3 className="font-medium">{analysis.filename}</h3>
                            <p className="text-sm text-gray-600 flex items-center space-x-1">
                              <Clock className="h-3 w-3" />
                              <span>{formatTimestamp(analysis.timestamp)}</span>
                            </p>
                          </div>
                          <div className="flex items-center space-x-4">
                            {getClassificationBadge(analysis.classification)}
                            <div className="text-right">
                              <div className="text-sm font-mono">
                                {(analysis.probability * 100).toFixed(1)}%
                              </div>
                              <div className={`text-xs font-mono ${getConfidenceColor(analysis.confidence)}`}>
                                {(analysis.confidence * 100).toFixed(1)}% conf
                              </div>
                            </div>
                          </div>
                        </div>
                        <Button 
                          variant="ghost" 
                          size="sm"
                          onClick={() => setSelectedAnalysis(analysis)}
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>
          </Tabs>

          {/* Detailed View Modal */}
          {selectedAnalysis && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <Card className="w-full max-w-2xl max-h-[80vh] overflow-y-auto">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle>{selectedAnalysis.filename}</CardTitle>
                      <CardDescription>
                        Analyzed on {formatTimestamp(selectedAnalysis.timestamp)}
                      </CardDescription>
                    </div>
                    <Button 
                      variant="ghost" 
                      size="sm"
                      onClick={() => setSelectedAnalysis(null)}
                    >
                      ×
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">Classification:</span>
                      {getClassificationBadge(selectedAnalysis.classification)}
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="font-medium">Probability:</span>
                      <span className="font-mono">
                        {(selectedAnalysis.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="font-medium">Confidence:</span>
                      <span className={`font-mono ${getConfidenceColor(selectedAnalysis.confidence)}`}>
                        {(selectedAnalysis.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      {selectedAnalysis.classification === 'Malignant'
                        ? 'This analysis indicated potential malignant tissue. Please consult with a medical professional.'
                        : 'This analysis indicated the tissue appears benign. Continue regular screening as recommended.'}
                    </AlertDescription>
                  </Alert>

                  <div className="space-y-2">
                    <p className="text-sm text-gray-600">
                      Analysis ID: {selectedAnalysis.id}
                    </p>
                    <p className="text-sm text-gray-600">
                      Timestamp: {selectedAnalysis.timestamp}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      )}
    </div>
  );
};