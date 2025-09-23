import React, { useState, useEffect } from 'react';
import { Button, Card, Table, Row, Col, Form, Badge, Alert } from 'react-bootstrap';
import { omrAPI } from '../services/api';
import { toast } from 'react-toastify';

const ResultsView = () => {
  const [omrData, setOmrData] = useState([]);
  const [answerKeys, setAnswerKeys] = useState([]);
  const [results, setResults] = useState([]);
  const [selectedOMR, setSelectedOMR] = useState('');
  const [selectedAnswerKey, setSelectedAnswerKey] = useState('');
  const [loading, setLoading] = useState(false);
  const [evaluating, setEvaluating] = useState(false);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [omrResponse, answerKeyResponse, resultsResponse] = await Promise.all([
        omrAPI.getOMRData(),
        omrAPI.getAnswerKeys(),
        omrAPI.getResults()
      ]);

      if (omrResponse.success) setOmrData(omrResponse.data);
      if (answerKeyResponse.success) setAnswerKeys(answerKeyResponse.data);
      if (resultsResponse.success) setResults(resultsResponse.data);
    } catch (error) {
      toast.error('Error fetching data: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleEvaluate = async () => {
    if (!selectedOMR || !selectedAnswerKey) {
      toast.error('Please select both OMR data and answer key');
      return;
    }

    setEvaluating(true);
    try {
      const response = await omrAPI.evaluateOMR(selectedOMR, selectedAnswerKey);
      
      if (response.success) {
        toast.success('Evaluation completed successfully!');
        fetchData(); // Refresh data
        setSelectedOMR('');
        setSelectedAnswerKey('');
      } else {
        toast.error(response.message || 'Evaluation failed');
      }
    } catch (error) {
      toast.error('Error during evaluation: ' + error.message);
    } finally {
      setEvaluating(false);
    }
  };

  const getGradeBadge = (percentage) => {
    if (percentage >= 90) return <Badge bg="success">A+</Badge>;
    if (percentage >= 80) return <Badge bg="primary">A</Badge>;
    if (percentage >= 70) return <Badge bg="info">B</Badge>;
    if (percentage >= 60) return <Badge bg="warning">C</Badge>;
    if (percentage >= 50) return <Badge bg="secondary">D</Badge>;
    return <Badge bg="danger">F</Badge>;
  };

  const renderEvaluationSection = () => (
    <Card className="mb-4">
      <Card.Header>
        <h5>Evaluate OMR</h5>
      </Card.Header>
      <Card.Body>
        <Row>
          <Col md={5}>
            <Form.Group className="mb-3">
              <Form.Label>Select OMR Data</Form.Label>
              <Form.Select
                value={selectedOMR}
                onChange={(e) => setSelectedOMR(e.target.value)}
              >
                <option value="">Choose OMR data...</option>
                {omrData.map((omr, index) => (
                  <option key={omr.file_id || index} value={omr.file_id}>
                    {omr.roll_number || 'Unknown Roll'} - {omr.subject} ({omr.branch}, {omr.class})
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
          </Col>
          <Col md={5}>
            <Form.Group className="mb-3">
              <Form.Label>Select Answer Key</Form.Label>
              <Form.Select
                value={selectedAnswerKey}
                onChange={(e) => setSelectedAnswerKey(e.target.value)}
              >
                <option value="">Choose answer key...</option>
                {answerKeys.map((key, index) => (
                  <option key={key.answer_key_id || index} value={key.answer_key_id}>
                    {key.subject} - Set {key.set_code} ({key.branch}, {key.class})
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
          </Col>
          <Col md={2}>
            <Form.Label>&nbsp;</Form.Label>
            <Button
              variant="primary"
              className="w-100"
              onClick={handleEvaluate}
              disabled={evaluating || !selectedOMR || !selectedAnswerKey}
            >
              {evaluating ? 'Evaluating...' : 'Evaluate'}
            </Button>
          </Col>
        </Row>
      </Card.Body>
    </Card>
  );

  const renderOMRDataSection = () => (
    <Card className="mb-4">
      <Card.Header>
        <h5>Uploaded OMR Data</h5>
      </Card.Header>
      <Card.Body>
        {omrData.length === 0 ? (
          <Alert variant="info">No OMR data found. Upload some OMR sheets first.</Alert>
        ) : (
          <Table responsive striped>
            <thead>
              <tr>
                <th>Roll Number</th>
                <th>Subject</th>
                <th>Branch</th>
                <th>Class</th>
                <th>Set Code</th>
                <th>Answers</th>
                <th>Upload Time</th>
                <th>Method</th>
              </tr>
            </thead>
            <tbody>
              {omrData.map((omr, index) => (
                <tr key={omr.file_id || index}>
                  <td>{omr.roll_number || 'N/A'}</td>
                  <td>{omr.subject}</td>
                  <td>{omr.branch}</td>
                  <td>{omr.class}</td>
                  <td>{omr.set_code || 'N/A'}</td>
                  <td>{omr.total_answers || 0}</td>
                  <td>{new Date(omr.upload_time).toLocaleDateString()}</td>
                  <td>
                    <Badge bg={omr.processing_method === 'auto_extraction' ? 'success' : 'warning'}>
                      {omr.processing_method === 'auto_extraction' ? 'Auto' : 'Manual'}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card.Body>
    </Card>
  );

  const renderAnswerKeysSection = () => (
    <Card className="mb-4">
      <Card.Header>
        <h5>Answer Keys</h5>
      </Card.Header>
      <Card.Body>
        {answerKeys.length === 0 ? (
          <Alert variant="info">No answer keys found. Upload answer keys first.</Alert>
        ) : (
          <Table responsive striped>
            <thead>
              <tr>
                <th>Subject</th>
                <th>Branch</th>
                <th>Class</th>
                <th>Set Code</th>
                <th>Total Questions</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {answerKeys.map((key, index) => (
                <tr key={key.answer_key_id || index}>
                  <td>{key.subject}</td>
                  <td>{key.branch}</td>
                  <td>{key.class}</td>
                  <td><Badge bg="primary">{key.set_code}</Badge></td>
                  <td>{key.total_questions}</td>
                  <td>{new Date(key.created_time).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card.Body>
    </Card>
  );

  const renderResultsSection = () => (
    <Card>
      <Card.Header>
        <h5>Evaluation Results</h5>
      </Card.Header>
      <Card.Body>
        {results.length === 0 ? (
          <Alert variant="info">No evaluation results found. Evaluate some OMR sheets first.</Alert>
        ) : (
          <Table responsive striped>
            <thead>
              <tr>
                <th>Roll Number</th>
                <th>Subject</th>
                <th>Branch</th>
                <th>Class</th>
                <th>Correct</th>
                <th>Wrong</th>
                <th>Blank</th>
                <th>Total Marks</th>
                <th>Percentage</th>
                <th>Grade</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {results.map((result, index) => (
                <tr key={result.evaluation_id || index}>
                  <td><strong>{result.roll_number}</strong></td>
                  <td>{result.subject}</td>
                  <td>{result.branch}</td>
                  <td>{result.class}</td>
                  <td><Badge bg="success">{result.correct_answers}</Badge></td>
                  <td><Badge bg="danger">{result.wrong_answers}</Badge></td>
                  <td><Badge bg="secondary">{result.blank_answers}</Badge></td>
                  <td><strong>{result.total_marks}</strong> / {result.max_possible_marks}</td>
                  <td><strong>{result.percentage}%</strong></td>
                  <td>{getGradeBadge(result.percentage)}</td>
                  <td>{new Date(result.evaluation_time).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card.Body>
    </Card>
  );

  if (loading) {
    return (
      <div className="container mt-4 text-center">
        <h2>Loading...</h2>
      </div>
    );
  }

  return (
    <div className="container mt-4">
      <h2>OMR Results & Evaluation</h2>
      
      {renderEvaluationSection()}
      {renderOMRDataSection()}
      {renderAnswerKeysSection()}
      {renderResultsSection()}
    </div>
  );
};

export default ResultsView;