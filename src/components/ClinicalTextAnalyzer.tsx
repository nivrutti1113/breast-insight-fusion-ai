
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { Badge } from "@/components/ui/badge";
import { BookmarkIcon, Check, FileText, Info, Sparkles, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface TextAnalysisResult {
  risk_factors: Array<{factor: string, weight: number}>;
  biomarkers: Array<{name: string, status: string, significance: number}>;
  key_findings: string[];
  recommendation: string;
  risk_score: number;
}

const SAMPLE_CLINICAL_TEXT = `Patient is a 52-year-old female presenting with a palpable mass in the upper outer quadrant of the right breast. Family history significant for breast cancer in mother at age 58 and maternal aunt at age 62. Patient reports mild intermittent pain in the area for the past 2 months. No nipple discharge or skin changes noted. Previous mammogram from 3 years ago was BIRADS-2. Physical examination confirms 1.5 cm firm, mobile mass at 10 o'clock position, 4 cm from nipple. No axillary lymphadenopathy. BRCA1/2 testing negative. Hormone replacement therapy for 5 years post-menopause.`;

const ClinicalTextAnalyzer: React.FC = () => {
  const [clinicalText, setClinicalText] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [textAnalysisResult, setTextAnalysisResult] = useState<TextAnalysisResult | null>(null);
  const { toast } = useToast();

  const loadSampleText = () => {
    setClinicalText(SAMPLE_CLINICAL_TEXT);
    setTextAnalysisResult(null);
  };

  const analyzeText = () => {
    if (!clinicalText.trim()) {
      toast({
        title: "Empty Text",
        description: "Please enter clinical text for analysis",
        variant: "destructive"
      });
      return;
    }

    setIsAnalyzing(true);
    
    // Simulate analysis with a timeout
    setTimeout(() => {
      // Mocked data for demo purposes
      const mockResult: TextAnalysisResult = {
        risk_factors: [
          { factor: "Family history of breast cancer", weight: 0.8 },
          { factor: "Age > 50", weight: 0.6 },
          { factor: "Previous hormone replacement therapy", weight: 0.5 },
          { factor: "Palpable breast mass", weight: 0.7 }
        ],
        biomarkers: [
          { name: "BRCA1/2", status: "Negative", significance: 0.3 },
          { name: "Tissue Density", status: "Moderate", significance: 0.5 },
          { name: "Tumor Size", status: "1.5 cm", significance: 0.7 }
        ],
        key_findings: [
          "1.5 cm firm, mobile mass detected in right breast",
          "Positive family history in first-degree relatives",
          "Recent onset of localized pain",
          "Previous mammogram was BIRADS-2"
        ],
        recommendation: "Recommend immediate diagnostic mammogram and ultrasound followed by possible biopsy based on findings.",
        risk_score: 0.65
      };
      
      setTextAnalysisResult(mockResult);
      setIsAnalyzing(false);
      
      toast({
        title: "Analysis Complete",
        description: "Clinical text analysis has been completed",
        variant: "default"
      });
    }, 3000);
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-medical-primary">Clinical Text Analysis</CardTitle>
        <CardDescription>Enter patient clinical information for AI analysis</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          <div className="flex justify-end">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={loadSampleText} 
              className="text-xs"
            >
              <FileText className="mr-1 h-3 w-3" />
              Load Sample Text
            </Button>
          </div>
          
          <Textarea 
            value={clinicalText}
            onChange={(e) => setClinicalText(e.target.value)}
            placeholder="Enter clinical notes, patient history, or genetic data here..."
            className="min-h-32"
          />
          
          <Button 
            onClick={analyzeText} 
            disabled={!clinicalText.trim() || isAnalyzing}
            className="w-full bg-medical-primary hover:bg-medical-primary/90"
          >
            {isAnalyzing ? (
              <>
                <div className="loader mr-2 h-4 w-4 border-2 rounded-full border-t-2"></div>
                Analyzing Clinical Data...
              </>
            ) : (
              'Analyze Clinical Text'
            )}
          </Button>
        </div>

        {textAnalysisResult && (
          <div className="mt-6 space-y-4">
            <div>
              <h3 className="font-medium text-sm text-gray-600 flex items-center mb-2">
                <Info className="h-4 w-4 mr-1" /> KEY FINDINGS
              </h3>
              <ul className="space-y-2">
                {textAnalysisResult.key_findings.map((finding, idx) => (
                  <li key={idx} className="text-sm flex gap-2">
                    <span className="text-medical-primary">•</span>
                    <span>{finding}</span>
                  </li>
                ))}
              </ul>
            </div>
            
            <div>
              <h3 className="font-medium text-sm text-gray-600 flex items-center mb-2">
                <BookmarkIcon className="h-4 w-4 mr-1" /> RISK FACTORS
              </h3>
              <div className="flex flex-wrap gap-2">
                {textAnalysisResult.risk_factors.map((factor, idx) => (
                  <Badge 
                    key={idx} 
                    variant="outline" 
                    className={cn(
                      "flex items-center gap-1.5 py-1", 
                      factor.weight > 0.7 ? "border-medical-danger text-medical-danger" : 
                      factor.weight > 0.5 ? "border-medical-warning text-medical-warning" : 
                      "border-medical-secondary text-medical-secondary"
                    )}
                  >
                    {factor.factor}
                    <span className="bg-gray-100 px-1 rounded-sm text-xs font-normal">
                      {Math.round(factor.weight * 10)}
                    </span>
                  </Badge>
                ))}
              </div>
            </div>
            
            <div>
              <h3 className="font-medium text-sm text-gray-600 flex items-center mb-2">
                <Sparkles className="h-4 w-4 mr-1" /> BIOMARKERS
              </h3>
              <div className="grid grid-cols-3 gap-2">
                {textAnalysisResult.biomarkers.map((marker, idx) => (
                  <div key={idx} className="border rounded-md p-2">
                    <div className="text-xs font-medium">{marker.name}</div>
                    <div className="flex justify-between items-center mt-1">
                      <span className={cn(
                        "text-xs",
                        marker.status.toLowerCase() === "positive" || 
                        parseFloat(marker.status) > 3 ? 
                        "text-medical-danger" : 
                        "text-gray-600"
                      )}>
                        {marker.status}
                      </span>
                      <span className="text-xs text-gray-400">
                        sig: {marker.significance.toFixed(1)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="bg-medical-accent p-3 rounded-md">
              <div className="flex justify-between items-center">
                <h3 className="font-medium text-sm">RECOMMENDATION</h3>
                <div className="flex items-center space-x-1">
                  <span className="text-xs text-gray-500">AI Confidence:</span>
                  <div className="flex items-center">
                    {[...Array(5)].map((_, i) => (
                      <div 
                        key={i}
                        className={cn(
                          "w-2 h-2 rounded-full mx-0.5", 
                          i < Math.round(textAnalysisResult.risk_score * 5) ? 
                          "bg-medical-primary" : "bg-gray-200"
                        )}
                      />
                    ))}
                  </div>
                </div>
              </div>
              <p className="text-sm mt-2">{textAnalysisResult.recommendation}</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ClinicalTextAnalyzer;
