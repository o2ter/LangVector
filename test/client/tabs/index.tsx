//
//  index.tsx
//
//  Copyright (c) 2021 - 2024 O2ter Limited. All rights reserved.
//

import _ from 'lodash';
import React from 'react';
import { View } from '@o2ter/react-ui';
import { useSearchParams } from 'react-router-dom';
import { TabProps, TabsProps } from './types';
import { DefaultTabbar } from './tabbar';
import { TabsContext } from './context';
import { useStableCallback } from 'sugax';

const flapMapChildren = (
  children: React.ReactNode
): React.ReactNode[] => _.flatten(React.Children.map(children, c => (
  React.isValidElement(c) && c.type === React.Fragment ?
    flapMapChildren(c.props.children) : c
)));

export const Tab = ({
  classes,
  eventKey,
  children,
  ...props
}: TabProps) => {
  const activeKey = React.useContext(TabsContext);
  if (typeof window === 'undefined') {
    return activeKey === eventKey && (
      <View
        classes={classes}
        {...props}
      >
        {children}
      </View>
    );
  }
  return (
    <View
      classes={activeKey === eventKey ? classes : 'd-none'}
      {...props}
    >
      {children}
    </View>
  );
}

export const Tabs = ({
  history = true,
  defaultActiveKey,
  activeKey,
  onSelect,
  renderTabbar = (tabs, { onSelect }) => <DefaultTabbar tabs={tabs} onSelect={onSelect} />,
  children
}: TabsProps) => {

  const tabs = _.compact(_.map(
    flapMapChildren(children),
    c => React.isValidElement(c) && c.type === Tab ? c.props as TabProps : undefined
  ));

  const [params] = useSearchParams();
  const _historyKey = params.get('tab');
  const _history = history && _.isString(_historyKey) && _.some(tabs, x => x.eventKey === _historyKey) ? _historyKey : undefined;

  const [_activeKey, setActiveKey] = React.useState(_history ?? defaultActiveKey ?? _.first(tabs)?.eventKey);

  const _onSelect = useStableCallback((eventKey: string) => {
    if (onSelect) onSelect(eventKey);
    setActiveKey(eventKey);
    if (history) {
      const search = new URLSearchParams(window.location.search);
      search.set('tab', eventKey);
      window.history.replaceState(window.history.state, '', `?${search.toString()}`);
    }
  });

  return (
    <TabsContext.Provider value={_.isString(activeKey) ? activeKey : _activeKey}>
      <View classes='flex-fill'>
        {renderTabbar(tabs, { onSelect: _onSelect })}
        {children}
      </View>
    </TabsContext.Provider>
  );
};
