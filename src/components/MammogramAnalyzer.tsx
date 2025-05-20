
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { AlertCircle } from "lucide-react";

interface AnalysisResult {
  probability: number;
  heatmapUrl: string;
  detectedAreas: Array<{x: number, y: number, width: number, height: number, confidence: number}>;
}

const MammogramAnalyzer: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    if (file) {
      if (!file.type.includes('image')) {
        toast({
          title: "Invalid file type",
          description: "Please upload an image file",
          variant: "destructive"
        });
        return;
      }

      setSelectedFile(file);
      const fileUrl = URL.createObjectURL(file);
      setPreviewUrl(fileUrl);
      setAnalysisResult(null);
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;
    
    setIsAnalyzing(true);
    
    // Simulate analysis with a timeout
    setTimeout(() => {
      // Mocked data for demo purposes
      const mockResult: AnalysisResult = {
        probability: Math.random() * 0.7, // Random probability between 0 and 0.7
        heatmapUrl: "/placeholder.svg", // Using placeholder as heatmap
        detectedAreas: [
          { x: 120, y: 100, width: 40, height: 40, confidence: 0.85 },
          { x: 200, y: 150, width: 30, height: 30, confidence: 0.72 }
        ]
      };
      
      setAnalysisResult(mockResult);
      setIsAnalyzing(false);
    }, 2500);
  };

  const renderHeatmapOverlay = () => {
    if (!analysisResult) return null;
    
    return (
      <div className="absolute inset-0 bg-gradient-to-br from-medical-accent via-medical-warning to-medical-danger opacity-60 mix-blend-multiply" 
           style={{ clipPath: "circle(40% at 60% 40%)" }} />
    );
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-medical-primary">Mammogram Analysis</CardTitle>
        <CardDescription>Upload a mammogram image for AI-based analysis</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-center w-full">
            <label htmlFor="mammogram-upload" className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 border-gray-300">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                {previewUrl ? (
                  <div className="relative w-full h-full flex items-center justify-center">
                    <img 
                      src={previewUrl} 
                      alt="Mammogram preview" 
                      className="max-h-56 object-contain" 
                    />
                    {analysisResult && renderHeatmapOverlay()}
                    {analysisResult && analysisResult.detectedAreas.map((area, index) => (
                      <div 
                        key={index}
                        className="absolute border-2 border-medical-danger rounded-md"
                        style={{
                          left: `${area.x}px`,
                          top: `${area.y}px`,
                          width: `${area.width}px`,
                          height: `${area.height}px`
                        }}
                      >
                        <div className="absolute -top-6 left-0 bg-medical-danger text-white text-xs px-1 py-0.5 rounded">
                          {Math.round(area.confidence * 100)}%
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <>
                    <svg className="w-8 h-8 mb-4 text-gray-500" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                      <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
                    </svg>
                    <p className="mb-2 text-sm text-gray-500"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                    <p className="text-xs text-gray-500">SVG, PNG, JPG or DICOM</p>
                  </>
                )}
              </div>
              <input 
                id="mammogram-upload" 
                type="file" 
                accept="image/*,.dcm" 
                className="hidden" 
                onChange={handleFileChange}
              />
            </label>
          </div>
          
          <Button 
            onClick={analyzeImage} 
            disabled={!selectedFile || isAnalyzing}
            className="w-full bg-medical-primary hover:bg-medical-primary/90"
          >
            {isAnalyzing ? (
              <>
                <div className="loader mr-2 h-4 w-4 border-2 rounded-full border-t-2"></div>
                Analyzing...
              </>
            ) : (
              'Analyze Mammogram'
            )}
          </Button>
        </div>

        {analysisResult && (
          <div className="mt-6">
            <Tabs defaultValue="results">
              <TabsList className="w-full">
                <TabsTrigger value="results" className="flex-1">Analysis Results</TabsTrigger>
                <TabsTrigger value="details" className="flex-1">Technical Details</TabsTrigger>
              </TabsList>
              <TabsContent value="results">
                <div className="p-4 border rounded-md mt-2">
                  <h3 className="font-medium mb-2">Cancer Risk Assessment</h3>
                  
                  <div className="relative pt-1">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-white bg-medical-primary">
                          {Math.round(analysisResult.probability * 100)}% Risk
                        </span>
                      </div>
                      <div className="text-right">
                        <span className="text-xs font-semibold inline-block">
                          {analysisResult.probability >= 0.5 ? 'High Risk' : 'Low to Moderate Risk'}
                        </span>
                      </div>
                    </div>
                    <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-gray-200 mt-2">
                      <div 
                        style={{ width: `${analysisResult.probability * 100}%` }}
                        className={`prediction-bar shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center ${
                          analysisResult.probability >= 0.7 ? 'bg-medical-danger' : 
                          analysisResult.probability >= 0.4 ? 'bg-medical-warning' : 
                          'bg-medical-success'
                        }`}
                      ></div>
                    </div>
                  </div>

                  <div className="flex items-start mt-4">
                    <AlertCircle className="h-5 w-5 text-medical-primary mr-2 flex-shrink-0 mt-0.5" />
                    <p className="text-sm text-gray-600">
                      This analysis is for demonstration purposes only and should not be used for medical diagnosis. 
                      Always consult with a healthcare professional.
                    </p>
                  </div>
                </div>
              </TabsContent>
              <TabsContent value="details">
                <div className="p-4 border rounded-md mt-2 space-y-3">
                  <div>
                    <h3 className="font-medium text-sm text-gray-600">MODEL</h3>
                    <p className="font-mono text-xs">Vision Transformer (ViT-B/16) fine-tuned on DDSM</p>
                  </div>
                  <div>
                    <h3 className="font-medium text-sm text-gray-600">CONFIDENCE</h3>
                    <p className="font-mono text-xs">{(0.8 + (Math.random() * 0.15)).toFixed(4)}</p>
                  </div>
                  <div>
                    <h3 className="font-medium text-sm text-gray-600">PROCESSING TIME</h3>
                    <p className="font-mono text-xs">{(0.5 + Math.random() * 2).toFixed(2)}s</p>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default MammogramAnalyzer;
