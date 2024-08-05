//
//  types.tsx
//
//  Copyright (c) 2021 - 2024 O2ter Limited. All rights reserved.
//

import _ from 'lodash';
import React from 'react';
import { View } from '@o2ter/react-ui';

export type TabProps = React.ComponentPropsWithoutRef<typeof View> & {
  eventKey: string;
  title: string;
};
export type TabsProps = React.PropsWithChildren<{
  history?: boolean;
  defaultActiveKey?: string;
  activeKey?: string;
  onSelect?: (eventKey: string) => void;
  renderTabbar?: (tabs: TabProps[], action: {
    onSelect: (eventKey: string) => void;
  }) => React.ReactNode;
}>;
