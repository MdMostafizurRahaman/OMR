import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button, Form, Card, Alert, Spinner, Row, Col } from 'react-bootstrap';
import { omrAPI } from '../services/api';
import { toast } from 'react-toastify';

const OMRUpload = () => {
  const [formData, setFormData] = useState({
    branch: '',
    class_name: '',
    subject: '',
    exam_date: '',
    set_code: ''
  });
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [extractedData, setExtractedData] = useState(null);
  const [manualAnswers, setManualAnswers] = useState({});
  const [showManualEntry, setShowManualEntry] = useState(false);

  const onDrop = (acceptedFiles) => {
    setFile(acceptedFiles[0]);
    setExtractedData(null);
    setShowManualEntry(false);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png'],
      'application/pdf': ['.pdf']
    },
    multiple: false
  });

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      toast.error('Please select a file to upload');
      return;
    }

    if (!formData.branch || !formData.class_name || !formData.subject || !formData.exam_date) {
      toast.error('Please fill in all required fields');
      return;
    }

    setLoading(true);
    
    try {
      const uploadFormData = new FormData();
      uploadFormData.append('file', file);
      uploadFormData.append('branch', formData.branch);
      uploadFormData.append('class_name', formData.class_name);
      uploadFormData.append('subject', formData.subject);
      uploadFormData.append('exam_date', formData.exam_date);
      uploadFormData.append('set_code', formData.set_code);

      const response = await omrAPI.uploadOMR(uploadFormData);
      
      if (response.success) {
        setExtractedData(response.data);
        toast.success('OMR processed successfully!');
        
        // If extraction failed, show manual entry option
        if (response.data.processing_method === 'manual_entry_required') {
          setShowManualEntry(true);
          // Initialize manual answers with empty values
          const emptyAnswers = {};
          for (let i = 1; i <= 100; i++) {
            emptyAnswers[i.toString()] = '';
          }
          setManualAnswers(emptyAnswers);
        }
      } else {
        toast.error(response.message || 'Failed to process OMR');
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Error uploading OMR: ' + (error.response?.data?.message || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleManualAnswerChange = (questionNum, answer) => {
    setManualAnswers({
      ...manualAnswers,
      [questionNum]: answer.toUpperCase()
    });
  };

  const saveManualAnswers = () => {
    if (extractedData) {
      const updatedData = {
        ...extractedData,
        answers: manualAnswers,
        total_answers: Object.keys(manualAnswers).filter(q => manualAnswers[q]).length,
        processing_method: 'manual_entry'
      };
      setExtractedData(updatedData);
      setShowManualEntry(false);
      toast.success('Manual answers saved successfully!');
    }
  };

  const renderManualEntry = () => {
    if (!showManualEntry) return null;

    return (
      <Card className="mt-4">
        <Card.Header>
          <h5>Manual Answer Entry</h5>
          <small className="text-muted">Automatic extraction failed. Please enter answers manually.</small>
        </Card.Header>
        <Card.Body>
          <Row>
            {Array.from({ length: 100 }, (_, i) => i + 1).map(questionNum => (
              <Col md={2} key={questionNum} className="mb-2">
                <Form.Group>
                  <Form.Label size="sm">{questionNum}</Form.Label>
                  <Form.Select
                    size="sm"
                    value={manualAnswers[questionNum.toString()] || ''}
                    onChange={(e) => handleManualAnswerChange(questionNum.toString(), e.target.value)}
                  >
                    <option value="">-</option>
                    <option value="A">A</option>
                    <option value="B">B</option>
                    <option value="C">C</option>
                    <option value="D">D</option>
                  </Form.Select>
                </Form.Group>
              </Col>
            ))}
          </Row>
          <Button variant="primary" onClick={saveManualAnswers} className="mt-3">
            Save Manual Answers
          </Button>
        </Card.Body>
      </Card>
    );
  };

  const renderExtractedData = () => {
    if (!extractedData) return null;

    return (
      <Card className="mt-4">
        <Card.Header>
          <h5>Extracted Data</h5>
        </Card.Header>
        <Card.Body>
          <Row>
            <Col md={6}>
              <p><strong>Roll Number:</strong> {extractedData.roll_number || 'Not detected'}</p>
              <p><strong>Set Code:</strong> {extractedData.set_code || 'Not detected'}</p>
              <p><strong>Total Answers:</strong> {extractedData.total_answers}</p>
              <p><strong>Processing Method:</strong> {extractedData.processing_method}</p>
            </Col>
            <Col md={6}>
              <p><strong>Branch:</strong> {extractedData.branch}</p>
              <p><strong>Class:</strong> {extractedData.class}</p>
              <p><strong>Subject:</strong> {extractedData.subject}</p>
              <p><strong>Exam Date:</strong> {extractedData.exam_date}</p>
            </Col>
          </Row>
          
          {extractedData.answers && Object.keys(extractedData.answers).length > 0 && (
            <div className="mt-3">
              <h6>Extracted Answers:</h6>
              <div className="border p-3" style={{ maxHeight: '200px', overflowY: 'auto' }}>
                {Object.entries(extractedData.answers).map(([question, answer]) => (
                  <span key={question} className="badge bg-secondary me-2 mb-1">
                    {question}: {answer}
                  </span>
                ))}
              </div>
            </div>
          )}
          
          {extractedData.processing_method === 'manual_entry_required' && (
            <Alert variant="warning" className="mt-3">
              Automatic extraction failed. You can enter answers manually using the form below.
            </Alert>
          )}
        </Card.Body>
      </Card>
    );
  };

  return (
    <div className="container mt-4">
      <h2>Upload OMR Answer Sheet</h2>
      
      <Form onSubmit={handleSubmit}>
        <Row>
          <Col md={6}>
            <Form.Group className="mb-3">
              <Form.Label>Branch *</Form.Label>
              <Form.Control
                type="text"
                name="branch"
                value={formData.branch}
                onChange={handleInputChange}
                placeholder="e.g., Dhaka, Chittagong"
                required
              />
            </Form.Group>
          </Col>
          <Col md={6}>
            <Form.Group className="mb-3">
              <Form.Label>Class *</Form.Label>
              <Form.Control
                type="text"
                name="class_name"
                value={formData.class_name}
                onChange={handleInputChange}
                placeholder="e.g., HSC 2024, SSC 2025"
                required
              />
            </Form.Group>
          </Col>
        </Row>

        <Row>
          <Col md={6}>
            <Form.Group className="mb-3">
              <Form.Label>Subject *</Form.Label>
              <Form.Control
                type="text"
                name="subject"
                value={formData.subject}
                onChange={handleInputChange}
                placeholder="e.g., Physics, Chemistry, Math"
                required
              />
            </Form.Group>
          </Col>
          <Col md={6}>
            <Form.Group className="mb-3">
              <Form.Label>Exam Date *</Form.Label>
              <Form.Control
                type="date"
                name="exam_date"
                value={formData.exam_date}
                onChange={handleInputChange}
                required
              />
            </Form.Group>
          </Col>
        </Row>

        <Form.Group className="mb-3">
          <Form.Label>Set Code (Optional)</Form.Label>
          <Form.Control
            type="text"
            name="set_code"
            value={formData.set_code}
            onChange={handleInputChange}
            placeholder="e.g., A, B, C, D"
            maxLength={1}
          />
        </Form.Group>

        <Card className="mb-3">
          <Card.Body>
            <div {...getRootProps()} className={`dropzone p-4 text-center border-2 border-dashed ${isDragActive ? 'border-primary bg-light' : 'border-secondary'}`} style={{ cursor: 'pointer' }}>
              <input {...getInputProps()} />
              {file ? (
                <div>
                  <p className="mb-1"><strong>Selected file:</strong> {file.name}</p>
                  <small className="text-muted">Click or drag to change file</small>
                </div>
              ) : (
                <div>
                  <p className="mb-1">Drag and drop an OMR image or PDF here, or click to select</p>
                  <small className="text-muted">Supports JPG, PNG, PDF files</small>
                </div>
              )}
            </div>
          </Card.Body>
        </Card>

        <Button 
          variant="primary" 
          type="submit" 
          disabled={loading || !file}
          className="w-100"
        >
          {loading ? (
            <>
              <Spinner animation="border" size="sm" className="me-2" />
              Processing OMR...
            </>
          ) : (
            'Upload and Process OMR'
          )}
        </Button>
      </Form>

      {renderExtractedData()}
      {renderManualEntry()}
    </div>
  );
};

export default OMRUpload;