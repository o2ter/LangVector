//
//  index.js
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
import { useCallback, useEffect, useMemo, useState } from 'frosty';
import { io } from 'socket.io-client';

const ChatBox = ({ classes, text, ...props }) => (
  <Text classes={['m-3 px-3 py-1 rounded', classes]} {...props}>{text}</Text>
);

export const Chatbot = () => {

  const socket = useMemo(() => io(), []);

  const [input, setInput] = useState('');
  const [state, setState] = useState({});
  const [userMessage, setUserMessage] = useState('');
  const [partial, setPartial] = useState('');

  useEffect(() => {

    socket.on('response', ({
      partial,
      message,
      responseText,
      ...state
    }) => {
      setUserMessage(partial ? message : '');
      setPartial(partial ? responseText : '');
      setState(v => ({ ...v, ...state }));
    });

    socket.emit('sync');

  }, []);

  const submit = useCallback(() => {
    socket.emit('msg', input);
    setInput('');
  }, [input]);

  const onchange = useCallback((key, value) => {
    socket.emit('sync', _.set({}, key, value));
  }, []);

  return (
    <div className='d-flex flex-row flex-fill'>
      <div className='d-flex flex-column w-50 border-right'>
        <div className='d-flex flex-column flex-fill position-relative'>
          <ScrollView classes='absolute-fill'>
            <Text>{state.raw}</Text>
          </ScrollView>
        </div>
        <div className='d-flex flex-column border-top'>
          <div className='row mb-3 px-3'>
            <div className='col'>status: {state.status}</div>
            <div className='col'>tokens: {state.tokens}</div>
            <div className='col'>context size: {state.contextSize}/{state.maxContextSize}</div>
          </div>
          <div className='row mb-3 px-3 gap-3'>
            <div className='col input-group align-items-center gap-1'>
              <label>max tokens</label>
              <input
                className='form-control'
                type='number'
                min={0} step={1}
                value={state.options?.maxTokens ?? ''}
                onChange={(e) => onchange('maxTokens', e.target.valueAsNumber)}
              />
            </div>
            <div className='col input-group align-items-center gap-1'>
              <label>temperature</label>
              <input
                className='form-control'
                type='number'
                min={0} step={0.1}
                value={state.options?.temperature ?? ''}
                onChange={(e) => onchange('temperature', e.target.valueAsNumber)}
              />
            </div>
          </div>
          <div className='row mb-3 px-3 gap-3'>
            <div className='col input-group align-items-center gap-1'>
              <label>frequency penalty</label>
              <input
                className='form-control'
                type='number'
                min={0} max={1} step={1}
                value={state.options?.repeatPenalty?.frequencyPenalty ?? ''}
                onChange={(e) => onchange('repeatPenalty.frequencyPenalty', e.target.valueAsNumber)}
              />
            </div>
            <div className='col input-group align-items-center gap-1'>
              <label>presence penalty</label>
              <input
                className='form-control'
                type='number'
                min={0} max={1} step={0.1}
                value={state.options?.repeatPenalty?.presencePenalty ?? ''}
                onChange={(e) => onchange('repeatPenalty.presencePenalty', e.target.valueAsNumber)}
              />
            </div>
          </div>
          <div className='row mb-3 px-3 gap-3'>
            <div className='col input-group align-items-center gap-1'>
              <label>minP</label>
              <input
                className='form-control'
                type='number'
                min={0} max={1} step={0.1}
                value={state.options?.minP ?? ''}
                onChange={(e) => onchange('minP', e.target.valueAsNumber)}
              />
            </div>
            <div className='col input-group align-items-center gap-1'>
              <label>topK</label>
              <input
                className='form-control'
                type='number'
                min={0} step={1}
                value={state.options?.topK ?? ''}
                onChange={(e) => onchange('topK', e.target.valueAsNumber)}
              />
            </div>
            <div className='col input-group align-items-center gap-1'>
              <label>topP</label>
              <input
                className='form-control'
                type='number'
                min={0} max={1} step={0.1}
                value={state.options?.topP ?? ''}
                onChange={(e) => onchange('topP', e.target.valueAsNumber)}
              />
            </div>
          </div>
        </div>
      </div>
      <div className='d-flex flex-column w-50'>
        <div className='d-flex flex-column flex-fill position-relative'>
          <ScrollView classes='absolute-fill'>
            {_.map(state.history, (x, i) => {
              switch (x.type) {
                case 'user':
                  return (
                    <ChatBox
                      key={i}
                      classes='bg-primary ml-auto'
                      style={{ color: 'white' }}
                      text={x.text}
                    />
                  );
                case 'model':
                  return (
                    <ChatBox
                      key={i}
                      classes='bg-light mr-auto'
                      text={_.map(_.filter(x.response, s => _.isString(s)), s => s.trim()).join('\n')}
                    />
                  );
                default: return null;
              }
            })}
            {!_.isEmpty(userMessage) && (
              <ChatBox
                classes='bg-primary ml-auto'
                style={{ color: 'white' }}
                text={userMessage}
              />
            )}
            {!_.isEmpty(partial) && (
              <ChatBox
                classes='bg-light mr-auto'
                text={partial}
              />
            )}
          </ScrollView>
        </div>
        <div className='d-flex flex-row p-2 gap-2 border-top'>
          <TextInput
            classes='flex-fill'
            value={input}
            onChangeText={setInput}
            onSubmitEditing={submit}
          />
          <Button title='Send' onPress={submit} />
        </div>
      </div>
    </div>
  );
};
