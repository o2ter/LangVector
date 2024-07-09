//
//  index.js
//
//  The MIT License
//  Copyright (c) 2021 - 2024 O2ter Limited. All rights reserved.
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
import React from 'react';
import { io } from 'socket.io-client';
import { Button, ScrollView, Text, TextInput } from '@o2ter/react-ui';
import './css/main.scss';

const ChatBox = ({ classes, text, ...props }) => (
  <Text classes={['m-3 px-3 py-1 rounded', classes]} {...props}>{text}</Text>
);

export default () => {

  const socket = React.useMemo(() => io(), []);

  const [input, setInput] = React.useState('');
  const [state, setState] = React.useState({});
  const [userMessage, setUserMessage] = React.useState('');
  const [partial, setPartial] = React.useState('');

  React.useEffect(() => {

    socket.emit('sync');

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

  }, []);

  const submit = React.useCallback(() => {
    socket.emit('msg', input);
    setInput('');
  }, [input]);

  return (
    <div className='d-flex flex-row flex-fill'>
      <div className='d-flex flex-column w-50 border-right'>
        <div className='d-flex flex-column flex-fill border-right position-relative'>
          <ScrollView classes='absolute-fill'>
            <Text>{state.raw}</Text>
          </ScrollView>
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
                      text={_.last(_.filter(x.response, s => _.isString(s)))}
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
