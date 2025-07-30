import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Calendar, 
  Filter, 
  Trash2, 
  Eye, 
  Edit3, 
  BarChart3, 
  Search,
  Download,
  RefreshCw
} from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Input } from '../components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '../components/ui/dialog';
import { Textarea } from '../components/ui/textarea';
import { toast } from 'sonner';
import { format } from 'date-fns';

interface PredictionResult {
  probability: number;
  classification: string;
  confidence: number;
}

interface ImageMetadata {
  filename: string;
  size: string;
  content_type: string;
  file_size: number;
}

interface ModelInfo {
  version: string;
  architecture: string;
  training_data: string;
}

interface PredictionHistory {
  id: string;
  prediction: PredictionResult;
  image_metadata: ImageMetadata;
  model_info: ModelInfo;
  created_at: string;
  updated_at: string;
  notes?: string;
  patient_id?: string;
  gradcam_path?: string;
}

interface Statistics {
  total_predictions: number;
  benign_count: number;
  malignant_count: number;
  recent_predictions: number;
  average_confidence: number;
  average_probability: number;
  benign_percentage: number;
  malignant_percentage: number;
}

const API_BASE_URL = 'http://localhost:8000';

export const HistoryPage: React.FC = () => {
  const [classificationFilter, setClassificationFilter] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedPrediction, setSelectedPrediction] = useState<PredictionHistory | null>(null);
  const [notes, setNotes] = useState<string>('');
  const [showNotesDialog, setShowNotesDialog] = useState(false);
  const [showStatsDialog, setShowStatsDialog] = useState(false);

  const queryClient = useQueryClient();

  // Fetch prediction history
  const { data: predictions, isLoading, error, refetch } = useQuery({
    queryKey: ['predictions', classificationFilter],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (classificationFilter) {
        params.append('classification', classificationFilter);
      }
      
      const response = await fetch(`${API_BASE_URL}/history?${params}`);
      if (!response.ok) throw new Error('Failed to fetch predictions');
      return response.json() as Promise<PredictionHistory[]>;
    },
  });

  // Fetch statistics
  const { data: statistics } = useQuery({
    queryKey: ['statistics'],
    queryFn: async () => {
      const response = await fetch(`${API_BASE_URL}/statistics`);
      if (!response.ok) throw new Error('Failed to fetch statistics');
      return response.json() as Promise<Statistics>;
    },
  });

  // Update notes mutation
  const updateNotesMutation = useMutation({
    mutationFn: async ({ id, notes }: { id: string; notes: string }) => {
      const response = await fetch(`${API_BASE_URL}/history/${id}/notes`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes }),
      });
      if (!response.ok) throw new Error('Failed to update notes');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['predictions'] });
      toast.success('Notes updated successfully');
      setShowNotesDialog(false);
    },
    onError: () => {
      toast.error('Failed to update notes');
    },
  });

  // Delete prediction mutation
  const deletePredictionMutation = useMutation({
    mutationFn: async (id: string) => {
      const response = await fetch(`${API_BASE_URL}/history/${id}`, {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Failed to delete prediction');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['predictions'] });
      queryClient.invalidateQueries({ queryKey: ['statistics'] });
      toast.success('Prediction deleted successfully');
    },
    onError: () => {
      toast.error('Failed to delete prediction');
    },
  });

  // Filter predictions based on search term
  const filteredPredictions = predictions?.filter(prediction =>
    prediction.image_metadata.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
    prediction.notes?.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  const handleUpdateNotes = () => {
    if (selectedPrediction) {
      updateNotesMutation.mutate({
        id: selectedPrediction.id,
        notes: notes,
      });
    }
  };

  const handleDeletePrediction = (id: string) => {
    if (confirm('Are you sure you want to delete this prediction?')) {
      deletePredictionMutation.mutate(id);
    }
  };

  const openNotesDialog = (prediction: PredictionHistory) => {
    setSelectedPrediction(prediction);
    setNotes(prediction.notes || '');
    setShowNotesDialog(true);
  };

  const getClassificationColor = (classification: string) => {
    return classification === 'Malignant' ? 'destructive' : 'secondary';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Prediction History</h1>
          <p className="text-gray-600 mt-2">View and manage your breast cancer prediction history</p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => setShowStatsDialog(true)}
            className="flex items-center gap-2"
          >
            <BarChart3 className="h-4 w-4" />
            Statistics
          </Button>
          <Button
            variant="outline"
            onClick={() => refetch()}
            className="flex items-center gap-2"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5" />
            Filters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4 flex-wrap">
            <div className="flex-1 min-w-[200px]">
              <Input
                placeholder="Search by filename or notes..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full"
              />
            </div>
            <Select value={classificationFilter} onValueChange={setClassificationFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="All Classifications" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">All Classifications</SelectItem>
                <SelectItem value="Benign">Benign</SelectItem>
                <SelectItem value="Malignant">Malignant</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Statistics Summary */}
      {statistics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold">{statistics.total_predictions}</div>
              <div className="text-sm text-gray-600">Total Predictions</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-green-600">{statistics.benign_count}</div>
              <div className="text-sm text-gray-600">Benign ({statistics.benign_percentage}%)</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-red-600">{statistics.malignant_count}</div>
              <div className="text-sm text-gray-600">Malignant ({statistics.malignant_percentage}%)</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold">{(statistics.average_confidence * 100).toFixed(1)}%</div>
              <div className="text-sm text-gray-600">Avg. Confidence</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Predictions List */}
      {isLoading ? (
        <div className="flex justify-center items-center py-12">
          <div className="text-gray-500">Loading predictions...</div>
        </div>
      ) : error ? (
        <Card>
          <CardContent className="p-8 text-center">
            <div className="text-red-500 mb-2">Error loading predictions</div>
            <Button onClick={() => refetch()} variant="outline">
              Try Again
            </Button>
          </CardContent>
        </Card>
      ) : filteredPredictions.length === 0 ? (
        <Card>
          <CardContent className="p-8 text-center">
            <div className="text-gray-500 mb-4">No predictions found</div>
            <Button onClick={() => window.location.href = '/analysis'}>
              Create First Prediction
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {filteredPredictions.map((prediction) => (
            <Card key={prediction.id} className="hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="flex justify-between items-start mb-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="font-semibold text-lg">{prediction.image_metadata.filename}</h3>
                      <Badge variant={getClassificationColor(prediction.prediction.classification)}>
                        {prediction.prediction.classification}
                      </Badge>
                      <span className={`text-sm font-medium ${getConfidenceColor(prediction.prediction.confidence)}`}>
                        {(prediction.prediction.confidence * 100).toFixed(1)}% confidence
                      </span>
                    </div>
                    <div className="text-sm text-gray-600 space-y-1">
                      <div>Probability: {(prediction.prediction.probability * 100).toFixed(2)}%</div>
                      <div>Size: {prediction.image_metadata.size}</div>
                      <div>Date: {format(new Date(prediction.created_at), 'PPp')}</div>
                      {prediction.notes && (
                        <div className="mt-2 p-2 bg-gray-50 rounded text-sm">
                          <strong>Notes:</strong> {prediction.notes}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => openNotesDialog(prediction)}
                    >
                      <Edit3 className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleDeletePrediction(prediction.id)}
                      className="text-red-600 hover:text-red-700"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Notes Dialog */}
      <Dialog open={showNotesDialog} onOpenChange={setShowNotesDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Notes</DialogTitle>
            <DialogDescription>
              Add or edit notes for this prediction.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <Textarea
              placeholder="Enter your notes here..."
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={4}
            />
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowNotesDialog(false)}>
                Cancel
              </Button>
              <Button 
                onClick={handleUpdateNotes}
                disabled={updateNotesMutation.isPending}
              >
                {updateNotesMutation.isPending ? 'Saving...' : 'Save Notes'}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Statistics Dialog */}
      <Dialog open={showStatsDialog} onOpenChange={setShowStatsDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Detailed Statistics</DialogTitle>
            <DialogDescription>
              Comprehensive overview of your prediction history.
            </DialogDescription>
          </DialogHeader>
          {statistics && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">{statistics.total_predictions}</div>
                  <div className="text-sm text-blue-800">Total Predictions</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">{statistics.recent_predictions}</div>
                  <div className="text-sm text-green-800">Recent (30 days)</div>
                </div>
              </div>
              
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-2">
                    <span>Benign Cases</span>
                    <span>{statistics.benign_count} ({statistics.benign_percentage}%)</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-600 h-2 rounded-full" 
                      style={{ width: `${statistics.benign_percentage}%` }}
                    ></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between mb-2">
                    <span>Malignant Cases</span>
                    <span>{statistics.malignant_count} ({statistics.malignant_percentage}%)</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-red-600 h-2 rounded-full" 
                      style={{ width: `${statistics.malignant_percentage}%` }}
                    ></div>
                  </div>
                </div>
              </div>
              
              <div className="pt-4 border-t">
                <div className="text-sm text-gray-600 space-y-2">
                  <div>Average Confidence: {(statistics.average_confidence * 100).toFixed(1)}%</div>
                  <div>Average Probability: {(statistics.average_probability * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};