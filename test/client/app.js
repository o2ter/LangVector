//
//  app.js
//
//  The MIT License
//  Copyright (c) 2021 - 2025 O2ter Limited. All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

import _ from 'lodash';
import { Chatbot } from './chatbot';
import { Similarity } from './similarity';
import './css/main.scss';
import 'bootstrap';

export const App = () => (
  <div className='d-flex flex-column flex-fill'>
    <ul className="nav nav-tabs" id="myTab" role="tablist">
      <li className="nav-item" role="presentation">
        <button className="nav-link active" id="chatbot-tab" data-bs-toggle="tab" data-bs-target="#chatbot" type="button" role="tab" aria-controls="chatbot" aria-selected="true">Chatbot</button>
      </li>
      <li className="nav-item" role="presentation">
        <button className="nav-link" id="similarity-tab" data-bs-toggle="tab" data-bs-target="#similarity" type="button" role="tab" aria-controls="similarity" aria-selected="false">Similarity</button>
      </li>
    </ul>
    <div className="tab-content" id="myTabContent">
      <div className="tab-pane fade show active" id="chatbot" role="tabpanel" aria-labelledby="chatbot-tab">
        {/* <Chatbot /> */}
      </div>
      <div className="tab-pane fade" id="similarity" role="tabpanel" aria-labelledby="similarity-tab">
        <Similarity />
      </div>
    </div>
  </div>
);