import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Navbar, Nav, Container } from 'react-bootstrap';
import { ToastContainer } from 'react-toastify';

import OMRUpload from './components/OMRUpload';
import AnswerKeyUpload from './components/AnswerKeyUpload';
import ResultsView from './components/ResultsView';

import 'bootstrap/dist/css/bootstrap.min.css';
import 'react-toastify/dist/ReactToastify.css';

const HomePage = () => (
  <Container className="mt-5">
    <div className="text-center">
      <h1>OMR Processing System</h1>
      <p className="lead">Automated OMR answer sheet processing for coaching centers</p>
      
      <div className="row mt-5">
        <div className="col-md-4">
          <div className="card h-100">
            <div className="card-body text-center">
              <h5>Upload OMR</h5>
              <p>Upload student answer sheets for automatic processing</p>
              <Link to="/upload-omr" className="btn btn-primary">
                Upload OMR Sheets
              </Link>
            </div>
          </div>
        </div>
        
        <div className="col-md-4">
          <div className="card h-100">
            <div className="card-body text-center">
              <h5>Answer Keys</h5>
              <p>Upload correct answers for question sets</p>
              <Link to="/answer-key" className="btn btn-success">
                Manage Answer Keys
              </Link>
            </div>
          </div>
        </div>
        
        <div className="col-md-4">
          <div className="card h-100">
            <div className="card-body text-center">
              <h5>Results</h5>
              <p>View evaluation results and student performance</p>
              <Link to="/results" className="btn btn-info">
                View Results
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  </Container>
);

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar bg="dark" variant="dark" expand="lg">
          <Container>
            <Navbar.Brand as={Link} to="/">
              OMR System
            </Navbar.Brand>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
              <Nav className="me-auto">
                <Nav.Link as={Link} to="/">
                  Home
                </Nav.Link>
                <Nav.Link as={Link} to="/upload-omr">
                  Upload OMR
                </Nav.Link>
                <Nav.Link as={Link} to="/answer-key">
                  Answer Keys
                </Nav.Link>
                <Nav.Link as={Link} to="/results">
                  Results
                </Nav.Link>
              </Nav>
            </Navbar.Collapse>
          </Container>
        </Navbar>

        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/upload-omr" element={<OMRUpload />} />
          <Route path="/answer-key" element={<AnswerKeyUpload />} />
          <Route path="/results" element={<ResultsView />} />
        </Routes>

        <ToastContainer
          position="top-right"
          autoClose={5000}
          hideProgressBar={false}
          newestOnTop={false}
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
        />
      </div>
    </Router>
  );
}

export default App;