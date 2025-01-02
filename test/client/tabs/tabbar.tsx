//
//  tabbar.tsx
//
//  Copyright (c) 2021 - 2025 O2ter Limited. All rights reserved.
//

import _ from 'lodash';
import React from 'react';
import { Text, Pressable, ScrollView, View } from '@o2ter/react-ui';
import { TabProps } from './types';
import { TabsContext } from './context';

type DefaultTabbarProps = {
  tabs: TabProps[];
  onSelect: (key: string) => void;
};
export const DefaultTabbar = ({
  tabs,
  onSelect,
}: DefaultTabbarProps) => {
  const activeKey = React.useContext(TabsContext);
  return (
      <View classes='flex-row border-bottom-1 max-w-100'>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
        >
          <View classes='flex-row gap-3'>
            {_.map(tabs, ({ eventKey, title }) => (
              <Pressable
                classes={[
                  activeKey === eventKey ? 'border-bottom-2 border-primary' : '',
                  'py-2',
                ]}
                key={eventKey}
                onPress={() => onSelect(eventKey)}
              >
                <Text
                  classes={[
                    activeKey === eventKey ? 'text-primary' : 'text-gray-700',
                    'fw-semibold'
                  ]}
                >{title}</Text>
              </Pressable>
            ))}
          </View>
        </ScrollView>
      </View>
  );
};
