import React, { useState } from 'react';
import { Button, Form, Card, Row, Col, Alert } from 'react-bootstrap';
import { omrAPI } from '../services/api';
import { toast } from 'react-toastify';

const AnswerKeyUpload = () => {
  const [formData, setFormData] = useState({
    branch: '',
    class_name: '',
    subject: '',
    set_code: '',
    total_questions: 100
  });
  
  const [answers, setAnswers] = useState({});
  const [loading, setLoading] = useState(false);

  // Initialize answers when total_questions changes
  React.useEffect(() => {
    const newAnswers = {};
    for (let i = 1; i <= formData.total_questions; i++) {
      newAnswers[i.toString()] = answers[i.toString()] || '';
    }
    setAnswers(newAnswers);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [formData.total_questions, answers]);

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleAnswerChange = (questionNum, answer) => {
    setAnswers({
      ...answers,
      [questionNum]: answer.toUpperCase()
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate required fields
    if (!formData.branch || !formData.class_name || !formData.subject || !formData.set_code) {
      toast.error('Please fill in all required fields');
      return;
    }

    // Check if all answers are provided
    const emptyAnswers = Object.entries(answers).filter(([_, answer]) => !answer.trim());
    if (emptyAnswers.length > 0) {
      toast.error(`Please provide answers for all ${formData.total_questions} questions`);
      return;
    }

    setLoading(true);

    try {
      const answerKeyData = {
        branch: formData.branch,
        class_name: formData.class_name,
        subject: formData.subject,
        set_code: formData.set_code,
        total_questions: parseInt(formData.total_questions),
        answers: answers
      };

      const response = await omrAPI.uploadAnswerKey(answerKeyData);
      
      if (response.success) {
        toast.success('Answer key saved successfully!');
        // Reset form
        setFormData({
          branch: '',
          class_name: '',
          subject: '',
          set_code: '',
          total_questions: 100
        });
        setAnswers({});
      } else {
        toast.error(response.message || 'Failed to save answer key');
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Error saving answer key: ' + (error.response?.data?.message || error.message));
    } finally {
      setLoading(false);
    }
  };

  const fillSampleAnswers = () => {
    const sampleAnswers = {};
    const options = ['A', 'B', 'C', 'D'];
    
    for (let i = 1; i <= formData.total_questions; i++) {
      sampleAnswers[i.toString()] = options[Math.floor(Math.random() * options.length)];
    }
    
    setAnswers(sampleAnswers);
    toast.info('Sample answers filled. Please review and modify as needed.');
  };

  const clearAllAnswers = () => {
    const emptyAnswers = {};
    for (let i = 1; i <= formData.total_questions; i++) {
      emptyAnswers[i.toString()] = '';
    }
    setAnswers(emptyAnswers);
    toast.info('All answers cleared');
  };

  return (
    <div className="container mt-4">
      <h2>Upload Answer Key</h2>
      
      <Form onSubmit={handleSubmit}>
        <Card className="mb-4">
          <Card.Header>
            <h5>Question Set Information</h5>
          </Card.Header>
          <Card.Body>
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
              <Col md={3}>
                <Form.Group className="mb-3">
                  <Form.Label>Set Code *</Form.Label>
                  <Form.Control
                    type="text"
                    name="set_code"
                    value={formData.set_code}
                    onChange={handleInputChange}
                    placeholder="A, B, C, D"
                    maxLength={1}
                    required
                  />
                </Form.Group>
              </Col>
              <Col md={3}>
                <Form.Group className="mb-3">
                  <Form.Label>Total Questions *</Form.Label>
                  <Form.Select
                    name="total_questions"
                    value={formData.total_questions}
                    onChange={handleInputChange}
                  >
                    <option value={30}>30 Questions</option>
                    <option value={45}>45 Questions</option>
                    <option value={60}>60 Questions</option>
                    <option value={100}>100 Questions</option>
                  </Form.Select>
                </Form.Group>
              </Col>
            </Row>
          </Card.Body>
        </Card>

        <Card className="mb-4">
          <Card.Header className="d-flex justify-content-between align-items-center">
            <h5>Answer Key</h5>
            <div>
              <Button 
                variant="outline-secondary" 
                size="sm" 
                onClick={fillSampleAnswers}
                className="me-2"
              >
                Fill Sample
              </Button>
              <Button 
                variant="outline-danger" 
                size="sm" 
                onClick={clearAllAnswers}
              >
                Clear All
              </Button>
            </div>
          </Card.Header>
          <Card.Body>
            <Alert variant="info">
              <small>
                Enter the correct answer (A, B, C, or D) for each question. 
                You can use the "Fill Sample" button to generate random answers for testing.
              </small>
            </Alert>
            
            <Row>
              {Array.from({ length: formData.total_questions }, (_, i) => i + 1).map(questionNum => (
                <Col md={2} lg={1} key={questionNum} className="mb-2">
                  <Form.Group>
                    <Form.Label size="sm" className="fw-bold">{questionNum}</Form.Label>
                    <Form.Select
                      size="sm"
                      value={answers[questionNum.toString()] || ''}
                      onChange={(e) => handleAnswerChange(questionNum.toString(), e.target.value)}
                      className={answers[questionNum.toString()] ? 'bg-light' : ''}
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
          </Card.Body>
        </Card>

        <div className="d-flex justify-content-between align-items-center mb-3">
          <div>
            <strong>Progress: </strong>
            {Object.values(answers).filter(a => a.trim()).length} / {formData.total_questions} questions completed
          </div>
          <Button 
            variant="primary" 
            type="submit" 
            disabled={loading || Object.values(answers).filter(a => a.trim()).length !== formData.total_questions}
            size="lg"
          >
            {loading ? 'Saving...' : 'Save Answer Key'}
          </Button>
        </div>
      </Form>
    </div>
  );
};

export default AnswerKeyUpload;