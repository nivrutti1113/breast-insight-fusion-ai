
import React, { useState } from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import MammogramAnalyzer from "@/components/MammogramAnalyzer";
import ClinicalTextAnalyzer from "@/components/ClinicalTextAnalyzer";
import FusionModel from "@/components/FusionModel";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { AlertCircle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

const Index: React.FC = () => {
  const [hasImageData, setHasImageData] = useState<boolean>(false);
  const [hasTextData, setHasTextData] = useState<boolean>(false);
  
  // These functions would be triggered by successful analysis in the child components
  const onImageAnalysisComplete = () => {
    setHasImageData(true);
  };
  
  const onTextAnalysisComplete = () => {
    setHasTextData(true);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Header />
      
      <main className="container mx-auto px-4 flex-grow">
        <Alert className="mb-6 bg-blue-50 border-blue-200">
          <AlertCircle className="h-4 w-4 text-blue-600" />
          <AlertTitle className="text-blue-600">Demonstration System</AlertTitle>
          <AlertDescription>
            This is a simulated medical AI platform for educational purposes. No real patient data or medical analysis is performed.
          </AlertDescription>
        </Alert>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Tabs defaultValue="image" className="w-full">
              <TabsList className="grid grid-cols-2 w-full mb-6">
                <TabsTrigger value="image">Image Analysis</TabsTrigger>
                <TabsTrigger value="text">Clinical Data Analysis</TabsTrigger>
              </TabsList>
              
              <TabsContent value="image" className="space-y-4">
                <MammogramAnalyzer />
                
                {/* Simplified mock output - in a real system this would be from actual component */}
                <Card className="hidden">
                  <CardHeader>
                    <CardTitle>Image Analysis Complete</CardTitle>
                    <CardDescription>Key findings from the mammogram analysis</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p>Analysis results would appear here.</p>
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="text" className="space-y-4">
                <ClinicalTextAnalyzer />
                
                {/* Simplified mock output - in a real system this would be from actual component */}
                <Card className="hidden">
                  <CardHeader>
                    <CardTitle>Text Analysis Complete</CardTitle>
                    <CardDescription>Key insights from clinical data</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p>Analysis results would appear here.</p>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
          
          <div>
            <FusionModel hasImage={true} hasText={true} />
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default Index;
