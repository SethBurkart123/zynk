import type { TaskWireCheck } from './generated/api';

const wireFixture = {
  kind: 'task_wire_check',
  priority: 'urgent',
  status: 'in_progress',
  numericStatus: 2,
} satisfies TaskWireCheck;

const literalKind: 'task_wire_check' = wireFixture.kind;
const priorityValue: 'low' | 'medium' | 'high' | 'urgent' = wireFixture.priority;
const statusValue: 'todo' | 'in_progress' | 'done' | 'cancelled' = wireFixture.status;
const numericStatusValue: 1 | 2 | 3 = wireFixture.numericStatus;

export const wireFidelityTypecheck = {
  literalKind,
  priorityValue,
  statusValue,
  numericStatusValue,
};
