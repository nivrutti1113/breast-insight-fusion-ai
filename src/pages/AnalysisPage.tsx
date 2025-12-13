import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation } from '@tanstack/react-query';
import { Upload, FileImage, AlertCircle, Download, Eye, Loader2 } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Alert, AlertDescription } from '../components/ui/alert';
import { Badge } from '../components/ui/badge';
import { Progress } from '../components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { toast } from 'sonner';

interface AnalysisResult {
  id: string;
  prediction: {
    probability: number;
    classification: string;
    confidence: number;
  };
  metadata: {
    filename: string;
    upload_time: string;
    model_version: string;
  };
  gradcam_overlay?: string;
}

export const AnalysisPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [progress, setProgress] = useState(0);
  const [showGradCam, setShowGradCam] = useState(false);

  const analysisMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Analysis failed');
      }
      
      return response.json();
    },
    onSuccess: (data) => {
      setAnalysisResult(data);
      setProgress(100);
      toast.success('Analysis completed successfully!');
    },
    onError: (error) => {
      toast.error('Analysis failed. Please try again.');
      console.error('Analysis error:', error);
    },
  });

  const reportMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('http://localhost:8000/analyze-and-report', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Report generation failed');
      }
      
      return response.blob();
    },
    onSuccess: (blob) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `breast_cancer_report_${new Date().toISOString().split('T')[0]}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success('Report downloaded successfully!');
    },
    onError: (error) => {
      toast.error('Report generation failed. Please try again.');
      console.error('Report error:', error);
    },
  });

  const gradcamMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('http://localhost:8000/gradcam', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Grad-CAM generation failed');
      }
      
      return response.blob();
    },
    onSuccess: (blob) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `gradcam_heatmap_${new Date().toISOString().split('T')[0]}.png`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success('Grad-CAM heatmap downloaded successfully!');
    },
    onError: (error) => {
      toast.error('Grad-CAM generation failed. Please try again.');
      console.error('Grad-CAM error:', error);
    },
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setAnalysisResult(null);
      setProgress(0);
      
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.dcm']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const handleAnalyze = () => {
    if (selectedFile) {
      setProgress(10);
      analysisMutation.mutate(selectedFile);
    }
  };

  const handleGenerateReport = () => {
    if (selectedFile) {
      reportMutation.mutate(selectedFile);
    }
  };

  const handleGenerateGradCAM = () => {
    if (selectedFile) {
      gradcamMutation.mutate(selectedFile);
    }
  };

  const getClassificationColor = (classification: string) => {
    return classification === 'Malignant' ? 'destructive' : 'secondary';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return 'text-green-600';
    if (confidence > 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-8">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold text-gray-900">Mammogram Analysis</h1>
        <p className="text-gray-600">
          Upload a mammogram image for AI-powered breast cancer detection
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Upload className="h-5 w-5" />
              <span>Upload Mammogram</span>
            </CardTitle>
            <CardDescription>
              Select a mammogram image (JPEG, PNG, or DICOM format, max 10MB)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
            >
              <input {...getInputProps()} />
              {imagePreview ? (
                <div className="space-y-4">
                  <img
                    src={imagePreview}
                    alt="Mammogram preview"
                    className="max-h-64 mx-auto rounded-lg shadow-md"
                  />
                  <p className="text-sm text-gray-600">
                    {selectedFile?.name}
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  <FileImage className="h-12 w-12 text-gray-400 mx-auto" />
                  <div>
                    <p className="text-lg font-medium">
                      {isDragActive ? 'Drop image here' : 'Drag & drop image here'}
                    </p>
                    <p className="text-sm text-gray-500">
                      or click to select file
                    </p>
                  </div>
                </div>
              )}
            </div>

            {selectedFile && (
              <div className="space-y-3">
                <Button
                  onClick={handleAnalyze}
                  disabled={analysisMutation.isPending}
                  className="w-full"
                >
                  {analysisMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Eye className="mr-2 h-4 w-4" />
                      Analyze Image
                    </>
                  )}
                </Button>

                {analysisMutation.isPending && (
                  <Progress value={progress} className="w-full" />
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card>
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
            <CardDescription>
              AI-powered analysis results and recommendations
            </CardDescription>
          </CardHeader>
          <CardContent>
            {analysisResult ? (
              <div className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Classification:</span>
                    <Badge variant={getClassificationColor(analysisResult.prediction.classification)}>
                      {analysisResult.prediction.classification}
                    </Badge>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Probability:</span>
                    <span className="font-mono">
                      {(analysisResult.prediction.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Confidence:</span>
                    <span className={`font-mono ${getConfidenceColor(analysisResult.prediction.confidence)}`}>
                      {(analysisResult.prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    {analysisResult.prediction.classification === 'Malignant'
                      ? 'The analysis suggests potential malignant tissue. Please consult with a medical professional immediately.'
                      : 'The analysis suggests the tissue appears benign. Continue regular screening as recommended.'}
                  </AlertDescription>
                </Alert>

                {/* Grad-CAM Overlay Display */}
                {analysisResult.gradcam_overlay && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium">Grad-CAM Heatmap</h4>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setShowGradCam(!showGradCam)}
                      >
                        <Eye className="h-4 w-4 mr-2" />
                        {showGradCam ? 'Hide' : 'Show'} Heatmap
                      </Button>
                    </div>
                    
                    {showGradCam && (
                      <div className="relative">
                        <img
                          src={`data:image/png;base64,${analysisResult.gradcam_overlay}`}
                          alt="Grad-CAM Heatmap Overlay"
                          className="w-full max-w-md mx-auto rounded-lg border shadow-sm"
                        />
                        <p className="text-sm text-gray-600 mt-2 text-center">
                          Red areas indicate regions the AI model focused on for its prediction
                        </p>
                      </div>
                    )}
                  </div>
                )}

                <div className="space-y-2">
                  <Button
                    onClick={handleGenerateReport}
                    disabled={reportMutation.isPending}
                    className="w-full"
                    variant="outline"
                  >
                    {reportMutation.isPending ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Generating Report...
                      </>
                    ) : (
                      <>
                        <Download className="mr-2 h-4 w-4" />
                        Download PDF Report
                      </>
                    )}
                  </Button>

                  <Button
                    onClick={handleGenerateGradCAM}
                    disabled={gradcamMutation.isPending}
                    className="w-full"
                    variant="outline"
                  >
                    {gradcamMutation.isPending ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Generating Heatmap...
                      </>
                    ) : (
                      <>
                        <Eye className="mr-2 h-4 w-4" />
                        Download Grad-CAM Heatmap
                      </>
                    )}
                  </Button>
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <FileImage className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Upload and analyze an image to see results</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Information Section */}
      <Card>
        <CardHeader>
          <CardTitle>Analysis Information</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="process" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="process">Process</TabsTrigger>
              <TabsTrigger value="gradcam">Grad-CAM</TabsTrigger>
              <TabsTrigger value="model">Model Info</TabsTrigger>
            </TabsList>
            
            <TabsContent value="process" className="space-y-4">
              <h3 className="text-lg font-semibold">Analysis Process</h3>
              <div className="space-y-2 text-sm text-gray-600">
                <p>1. Image preprocessing and normalization</p>
                <p>2. Feature extraction using ResNet50 architecture</p>
                <p>3. Binary classification (Benign/Malignant)</p>
                <p>4. Confidence score calculation</p>
                <p>5. Grad-CAM visualization generation</p>
              </div>
            </TabsContent>
            
            <TabsContent value="gradcam" className="space-y-4">
              <h3 className="text-lg font-semibold">Grad-CAM Visualization</h3>
              <div className="space-y-2 text-sm text-gray-600">
                <p>Gradient-weighted Class Activation Mapping (Grad-CAM) provides visual explanations for the model's decisions.</p>
                <p>• Red/Yellow areas: High model attention</p>
                <p>• Blue/Green areas: Low model attention</p>
                <p>• Helps identify suspicious regions in the mammogram</p>
              </div>
            </TabsContent>
            
            <TabsContent value="model" className="space-y-4">
              <h3 className="text-lg font-semibold">Model Information</h3>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• Architecture: ResNet50 with transfer learning</p>
                <p>• Training: Deep learning on mammography dataset</p>
                <p>• Input: 224x224 RGB images</p>
                <p>• Output: Binary classification with confidence score</p>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};